#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.
#import os
from pathlib import Path
import nltk
import spacy
import pandas as pd
import os
from nltk.tokenize import word_tokenize
import string
from readability import Readability
import cmudict
import pronouncing
import csv
from spacy.tokens import Doc
from collections import Counter

#path = Path.cwd() / "datafiles" / "novels"
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    formatted_text = formatted_text.replace("  ", "")
    formatted_text = text.replace("\n", " ")
    read = Readability(formatted_text)
    fk = read.flesch_kincaid()
    gradelevel = fk.grade_level
    return gradelevel


def vowelandycounter(word):
    counter = 0
    vowelandylist = ["a","e","i","o","u","y"]
    letters_in_word = list(word)
    for letter in letters_in_word:
        if str(letter).lower() in vowelandylist:
            counter +=1
    return counter


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters - 
    I have included y as a vowel in this estimate as it improved the estimation with the
    admittedly low number of words I tested it on. 

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    if word in d:
        return d[word]
    else:
        phones = pronouncing.phones_for_word(word)
        if phones !=[]:
            return pronouncing.syllable_count(phones[0])
        else: 
            return(vowelandycounter(word))

    pass

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    df= pd.DataFrame()
    pd.set_option('display.max_colwidth', 10000)
    for item in os.listdir(path):
        filepath = (str(path) +"/"+ str(item))
        text = pd.read_csv(filepath, sep='delimiter', header=None, engine = 'python', skip_blank_lines= True, quoting=csv.QUOTE_NONE, on_bad_lines = 'skip')
        fulltext = ""
        for row in text[0]:
            fulltext = fulltext + row
        item = item.strip(".txt")
        paramlist = item.split("-")
        itemdict = {"title" : paramlist[0], "author" : paramlist[1], "year": paramlist[2], "text" : fulltext}
        df = df._append(itemdict, ignore_index = True)
        df = df.sort_values(by = ["year"], ignore_index = True)
    return df


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    parseddocs = []
    nlp = spacy.load("en_core_web_sm") #Loads english language for spacy
    for row in df.iterrows():
        parsed = nlp(str(row[1]["text"][:99990]))
        parseddocs.append(parsed)
    new_column = {"Parsed Doc": parseddocs}
    df = df.assign(**new_column)
    df.to_pickle("./parsed.pickle")
    return df
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""

def nltk_ttr(text):
    #nltk.download('punkt')
    #nltk.download('punkt_tab')
    text = text.translate(str.maketrans('', '',string.punctuation))
    text = text.translate(str.maketrans('', '',string.digits))
    output = word_tokenize(text)
    unique_words = set()
    book_word_count = len(output)
    for word in output:
        unique_words.add(word.lower())
    unique_word_count = len(unique_words)
    ttr = (book_word_count / unique_word_count)
    return ttr
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""

def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[(row["title"])] = nltk_ttr(str(row["text"]))
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    resultslist = []
    for i, row in df.iterrows():
        fk_gl = round (float(fk_level(str(row["text"]), cmudict)), 4)
        resultslist.append(fk_gl)
        results["fk_level"] = resultslist
    df = df.assign(**results)
    return df


def subjects_by_verb_pmi(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    results = []
    counter = 0
    verbcounter = 0
    noun_counter =0
    doclength = len(doc["Parsed Doc"])
    for row in doc['title']:
        results.append([row])
    for row in doc['Parsed Doc']:
        for token in row:
            if token.lemma_ == verb:
                verbcounter +=1
        noun_counter = Counter((str(token))for token in row)
        noun_verb_counter = Counter((str(token)) for token in row if token.dep_ == "nsubj" and (token.head.lemma_ == verb ))
        pmi = dict()
        for key in noun_verb_counter.keys():
            pmi[key] = (noun_verb_counter[key] / doclength)/ (noun_counter[key] / doclength * verbcounter / doclength)
            pmi.update({key : pmi[key]})
        sorted_pmi_by_value = sorted(pmi.items(), key=lambda x:x[1])
        itemlist = []
        for item in sorted_pmi_by_value[-10:]:
            itemlist.append(item[0])
        results[counter].append(itemlist)
        counter +=1
    return results




def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    results = []
    counter = 0
    for row in doc['title']:
        results.append([row])
    for row in doc['Parsed Doc']:
        noun_counter = Counter((str(token)) for token in row if token.dep_ == "nsubj" and (token.head.lemma_ == verb ))
        sorted_nouncounter_by_value = sorted(noun_counter.items(), key=lambda x:x[1])
        itemlist = []
        for item in sorted_nouncounter_by_value[-10:]:
            itemlist.append(item[0])
        results[counter].append(itemlist)
        counter +=1
    return results




def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    resdict = {}
    wordlist = []
    for row in doc['Parsed Doc']:
        for token in row:
            if token.pos_ == "ADJ":
                if (str(token).lower()) not in wordlist:
                    resdict[(str(token)).lower()] = 1
                    wordlist.append(str(token).lower())
                elif str(token) in wordlist:
                    resdict[str(token)] +=1
    sorted_resdict_by_value = sorted(resdict.items(), key=lambda x:x[1])
    results = []
    for i in range (-10,-1):
        results.append(sorted_resdict_by_value[i])
    return results

if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "datafiles" / "novels"
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #nltk.download("cmudict")
    #get_ttrs(df)
    #get_fks(df)
    df = parse(df)
    #nltk_ttr("Example of a sentence to be tokenized")
    #df = pd.read_pickle(Path.cwd()/"parsed.pickle")
    #print(adjective_counts(df))
    #print(subjects_by_verb_count(df, "hear"))
    #print(adjective_counts(df))
    print(subjects_by_verb_pmi(df, "hear"))

        

