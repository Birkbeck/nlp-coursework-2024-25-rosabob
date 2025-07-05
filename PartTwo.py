from pathlib import Path
import pandas as pd
import os
import csv

path = Path.cwd() / "datafiles" / "speeches" / "hansard40000.csv"


def read_speeches(path):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    df = pd.read_csv(path, sep=',', engine = 'python', header = 0)
    return df

if __name__ == "__main__":
    df = read_speeches(path)
    df = df.replace("Labour (Co-op)", "Labour")
    #print (df["party"].value_counts())
    # Therefore the most popular parties are Conservative, Labour, SNP, (Speaker - doesn't count as isn't a party), Lib Dem. 
    df = df.drop(df[(df['party'] != "Conservative") & (df['party'] != "Labour") & (df['party'] != "Scottish National Party") & (df['party'] != "Liberal Democrat")].index)
    #print (df["party"].value_counts())

