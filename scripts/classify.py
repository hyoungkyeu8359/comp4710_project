import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import os.path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import re
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

USE_SVM = False
USE_NAIVE_BAIYES = True
USE_RANDOM_FOREST = False

svm_path = './../trained-model/svm-model.sav'
naive_bayes_path = './../trained-model/naive-baiyes-model.sav'
random_forest_path = './../trained-model/random-forest-model.sav'

# svm_vectorizer_path = './../trained-model/vectorizer/svm-vectorizer.sav'
# naive_bayes_vectorizer_path = './../trained-model/vectorizer/naive-baiyes-vectorizer.sav'
# random_forest_vectorizer_path = './../trained-model/vectorizer/random-forest-vectorizer.sav'

df_path = "./../swcwang-final-dataset/tweets_combined_labeled1.csv"
df = pd.read_csv(df_path)

tweets = df.iloc[:, 0].values
labels = df.iloc[:, 1].values

print("Dataset size: ", tweets.size)

# preprocess data to clean it
tweets_dataset = []
for sentence in range(0, len(tweets)):
    # Remove all the special characters
    processed_tweet = re.sub(r'\W', ' ', str(tweets[sentence]))

    # remove all single characters
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)

    # Remove single characters from the start
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet)

    # Substituting multiple spaces with single space
    processed_tweet = re.sub(r'\s+', ' ', processed_tweet, flags=re.I)

    # Removing prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)

    # Converting to Lowercase
    processed_tweet = processed_tweet.lower()

    tweets_dataset.append(processed_tweet)

# ================== Classification Code start here ==================
stopwords_ = stopwords.words('english')
stopwords_.extend(["im", "ive"])
STOPWORDS = set(stopwords_)

# This is getting the features using tf-idf
MIN_DF = 10  # min # fords occurence
MAX_DF = 0.8  # max occurence (percentage) in the documents
MAX_FEATURES = 2500  # most frequently occurring words

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES, min_df=MIN_DF, max_df=MAX_DF, stop_words=STOPWORDS)
tfidf = vectorizer.fit(tweets_dataset)
vectorized_tweets = vectorizer.fit_transform(tweets_dataset).toarray()

D_START = 140
D_END = 160

# classification using SVM
if USE_SVM:
    svm_model = pickle.load(open(svm_path, 'rb'))
    predict = svm_model.predict(vectorized_tweets)
    print("\nSVM Prediciton: ")
    depressed_tweets = np.asarray(np.where(predict == 1))
    print(depressed_tweets.size)
    # score = svm_model.score(vectorized_tweets, labels)
    # print(score)


# classification using Naive Bayes
if USE_NAIVE_BAIYES:
    naive_bayes_model = pickle.load(open(naive_bayes_path, 'rb'))
    predict = naive_bayes_model.predict(vectorized_tweets)
    print("\nNaive Bayes Prediciton: ")
    depressed_tweets = np.asarray(np.where(predict == 1))
    print(depressed_tweets.size)
    # score = naive_bayes_model.score(vectorized_tweets, labels)
    # print(score)


# classification using Random Forest
if USE_RANDOM_FOREST:
    random_forest__model = pickle.load(open(random_forest_path, 'rb'))
    predict = random_forest__model.predict(vectorized_tweets)
    print("\nRandom Forest Prediciton: ")
    depressed_tweets = np.asarray(np.where(predict == 1))
    print(depressed_tweets.size)
    # score = random_forest__model.score(vectorized_tweets, labels)
    # print(score)
