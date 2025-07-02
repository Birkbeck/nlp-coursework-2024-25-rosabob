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

#print("cwd is", os.getcwd())
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
    read = Readability(text)
    fk = flesch-kincaid()
    gradelevel = fk.grade_level
    return gradelevel

    pass

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
        text = pd.read_csv(filepath, sep='delimiter', header=None, engine = 'python')
        item = item.strip(".txt")
        paramlist = item.split("-")
        itemdict = {"title" : paramlist[0], "author" : paramlist[1], "year": paramlist[2], "text" : text}
        df = df._append(itemdict, ignore_index = True)
        df = df.sort_values(by = ["year"], ignore_index = True)
    return df


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    parseddocs = []
    nlp = spacy.load("en_core_web_sm") #Loads english language for spacy 
    for row in df.iterrows():
        parsed = nlp(str(row[1]["text"]))
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
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    df = df.assign(results)


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass

sylldict = {}

if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "datafiles" / "novels"
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    nltk.download("cmudict")
    df = parse(df)
    #nltk_ttr("Example of a sentence to be tokenized")
    get_ttrs(df)
    #print(df.head())
    print(count_syl("artificiality", sylldict))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

