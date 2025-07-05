from pathlib import Path
import pandas as pd
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.model_selection import train_test_split
import numpy as np 


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




if __name__ == "__main__":
    df = read_speeches(path)
    df = clean_df(df)
    vectorised_results = vectorise(df)
    vr_train, vr_test = train_test_split(vectorised_results, test_size=0.2, random_state=26)
    print(vr_train)


