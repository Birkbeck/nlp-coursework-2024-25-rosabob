from pathlib import Path
import pandas as pd
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 



path = Path.cwd() / "datafiles" / "speeches" / "hansard40000.csv"


def read_speeches(path):
    """Reads speeches from csv and puts into pd df"""
    df = pd.read_csv(path, sep=',', engine = 'python', header = 0)
    return df

def clean_df(df):
    """Performs cleaning tasks required in question a"""
    df = df.replace("Labour (Co-op)", "Labour")
    #print (df["party"].value_counts())
    # Therefore the most popular parties are Conservative, Labour, SNP, (Speaker - doesn't count as isn't a party), Lib Dem. 
    df = df.drop(df[(df['party'] != "Conservative") & (df['party'] != "Labour") & (df['party'] != "Scottish National Party") & (df['party'] != "Liberal Democrat")].index)
    #print (df["party"].value_counts())
    df = df.drop(df[(df['speech_class'] != "Speech")].index)
    #print (df["speech_class"].value_counts())
    lenlist =[]
    for row in df.iterrows():
        speech = row[1]["speech"]
        lenlist.append(len(speech))
    df["speech_length"] = lenlist
    df = df.drop(df[(df['speech_length'] < 1000)].index)
    #print (df["speech_length"].value_counts())
    print("The shape of the cleaned df is", df.shape)
    return df

def vectorise(df):
    """vectorises as required in question b"""
    tfidf = TfidfVectorizer(tokenizer = nltk.word_tokenize, max_features= 3000, stop_words='english')
    vect = tfidf.fit_transform(df["speech"])
    return vect

def vectorise_part_d(df):
    """vectorises as required in question d including unigrams, bigrams and trigrams"""
    tfidf = TfidfVectorizer(tokenizer = nltk.word_tokenize, max_features= 3000, stop_words='english', ngram_range=(1, 3))
    vect = tfidf.fit_transform(df["speech"])
    return vect

def vectorise_custom(df):
    """vectorises using my custom tokenizer"""
    tfidf = TfidfVectorizer(tokenizer = custom_tokenizer, max_features= 3000, stop_words='english', ngram_range=(1, 3))
    vect = tfidf.fit_transform(df["speech"])
    return vect

def test_train_split(vectors, df):
    vr_train, vr_test, party_train, party_test = train_test_split(vectors, df["party"], test_size=0.2, random_state=26, stratify = df["party"])
    return vr_train, vr_test, party_train, party_test

def random_forest(vr_train, vr_test, party_train, party_test):
    rf = RandomForestClassifier(n_estimators= 300)
    rf.fit(vr_train, party_train)
    party_pred = rf.predict(vr_test)
    accuracy = accuracy_score(party_test, party_pred)
    rf_f1 = f1_score(party_test, party_pred, average='macro')
    rf_classification = (classification_report(party_test, party_pred))
    return rf_f1, rf_classification

def support_vector(vr_train, vr_test, party_train, party_test):
    svmclf = svm.SVC(kernel = "linear")
    svmclf.fit(vr_train, party_train)
    party_pred =svmclf.predict(vr_test)
    accuracy = accuracy_score(party_test, party_pred)
    svm_f1 = f1_score(party_test, party_pred, average='macro')
    svm_classification = (classification_report(party_test, party_pred))
    return svm_f1, svm_classification

def custom_tokenizer(text):
    tokenlist = text.lower().split(" ")
    en_stop_words = set(stopwords.words('english'))
    return [token for token in tokenlist if token.isalpha() and token not in en_stop_words and len(token) >4]



if __name__ == "__main__":
    df = read_speeches(path)
    df = clean_df(df)
    vectorised_results = vectorise(df)
    vr_train, vr_test, party_train, party_test = test_train_split (vectorised_results, df)
    print ("The random forest f1 scores and classification reports are", random_forest(vr_train, vr_test, party_train, party_test))
    print ("The svm f1 scores and classification reports are", support_vector(vr_train, vr_test, party_train, party_test))
    vectorised_results_part_d = vectorise_part_d(df)
    vr_train_d, vr_test_d, party_train_d, party_test_d = test_train_split (vectorised_results_part_d, df)
    print ("The random forest f1 scores and classification reports for part d are", random_forest(vr_train_d, vr_test_d, party_train_d, party_test_d))
    print ("The svm f1 scores and classification reports for part d are", support_vector(vr_train_d, vr_test_d, party_train_d, party_test_d))
    vectorised_results_custom = vectorise_custom(df)
    vr_train_custom, vr_test_custom, party_train_custom, party_test_custom = test_train_split (vectorised_results_custom, df)
    print ("The random forest f1 scores and classification reports for my custom function are", random_forest(vr_train_custom, vr_test_custom, party_train_custom, party_test_custom))
    print ("The svm f1 scores and classification reports for my custom function are", support_vector(vr_train_custom, vr_test_custom, party_train_custom, party_test_custom))
    #for item in df["speech"]:
        #print(custom_tokenizer(item))
        #print(item)
        #exit()




