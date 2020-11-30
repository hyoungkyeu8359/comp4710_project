import pandas as pd
import numpy as np
import pathlib
import sys
import string
import preprocessor as p
import os
import shutil
import nltk
from collections import Counter
from slang_script import translator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from spellchecker import SpellChecker
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer



#Current directory
current_path = "../" 
newDir = "swcwang-final-dataset/"

# Following five functions are taken from https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing.
# They are used as a utility to pre process data for stop words, frequent, rare and lemmatize words
# Author sudalairajkumar
def remove_stopwords(text):
    # first_pronouns = ["i", "me", "my", "mine", "our", "ours", "us", "we"]
    # stopwords_filtered = [x for x in stopwords.words('english') if x not in first_pronouns]
    # STOPWORDS = set(stopwords_filtered)

    # Original stopwords without First person pronouns filtered
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def remove_freqwords(text):
    cnt = Counter()
    FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

def remove_rarewords(text):
    cnt = Counter()
    n_rare_words = 10
    RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

# Author of this function is sudalairajkumar and algorithm for spell checking by Peter Norvig
def correct_spellings(text):
    spell = SpellChecker()
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

#datasets to preprocess and clean 
csv_files = [1, 2, 3, 10, 16, 23, 30, 37, 128, 135, 142, 149, 156, 163,170, 177, 184,191, 198, 205 ] 

#Put them in this new directory. Delete the directory if it exists already
all_files = ["tweets_combined_target_1", "tweets_combined_target_0", "tweets_combined"]
shutil.rmtree(current_path + newDir, ignore_errors=True)
os.makedirs(current_path + newDir)

target_column = "tweet_processed"
#For each csv file do cleaning on the data texts
for file in all_files:
    print("Pre-processing for "+file+".csv")
    df = pd.read_csv(current_path + 'swcwang-combined-dataset/' + file + '.csv',lineterminator='\n')
    print('Starting cleaning')
    df[target_column] =  df["tweet"].map(lambda x: p.clean(str(x)))
    print('Finished cleaning. Starting lowercase')
    df[target_column] =  df[target_column].map(lambda x: x.lower())
    print('Finished lowercase. Starting punctation')
    df[target_column] =  df[target_column].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
    print('Finished punctuation. Starting acronym')
    df[target_column] =  df[target_column].map(lambda x: translator(str(x)))
    print('Finished acronym. Starting strip')
    df[target_column] =  df[target_column].map(lambda x: x.strip())
    # print('Finished strip. Starting stop words')
    # df[target_column] =  df[target_column].map(lambda x: remove_stopwords(x))
    # print('Finished stopwords. Starting freqwords')
    # df[target_column] = df[target_column].map(lambda x: remove_freqwords(x))
    # print('Finished freqwords. Starting rare words')
    # df[target_column] = df[target_column].map(lambda x: remove_rarewords(x))
    # WE COULD USE THIS BUT IT TAKES SO LONG!
    # df["text1"] = df["text1"].apply(lambda x: correct_spellings(x))
    print('Finished rare words. Starting lemmatize')
    df[target_column] = df[target_column].map(lambda x: lemmatize_words(x))
    # print('Finished lemmatize word. Starting tokenization')
    # df["text1"] = df["text1"].map(lambda x: word_tokenize(x))
    # print('Finished tokenization')
    df.to_csv(current_path + newDir + file + '.csv', index=False)

