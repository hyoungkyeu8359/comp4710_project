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

USE_SVM = True
USE_NAIVE_BAIYES = False
USE_RANDOM_FOREST = False

svm_path = './../trained-model/svm-model.sav'
naive_bayes_path = './../trained-model/naive-baiyes-model-1.sav'
random_forest_path = './../trained-model/random-forest-model.sav'

# svm_vectorizer_path = './../trained-model/vectorizer/svm-vectorizer.sav'
# naive_bayes_vectorizer_path = './../trained-model/vectorizer/naive-baiyes-vectorizer.sav'
# random_forest_vectorizer_path = './../trained-model/vectorizer/random-forest-vectorizer.sav'

file_num = str(1)
df_path = "./../cleaned-data/cleaned-data-"+file_num+".csv"
df = pd.read_csv(df_path, lineterminator='\n')

original_texts = df['text'].values
tweets = df['tweet_processed'].values


# preprocess data to clean it
tweets_dataset = []
tweets_dataset_index = []
tweets_text = []

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

    if processed_tweet not in tweets_dataset:
        tweets_dataset.append(processed_tweet)
        tweets_dataset_index.append(sentence)
        tweets_text.append(original_texts[sentence])

tweets_dataset = np.array(tweets_dataset)
tweets_text = np.array(tweets_text)
print("tweets_dataset: ", tweets_dataset.size)
print("tweets_text: ", tweets_text.size)

# print(tweets_dataset[17])
# print(tweets_text[17])
# exit()
# ================== Classification Code start here ==================
stopwords_ = stopwords.words('english')
stopwords_.extend(["im", "ive"])
STOPWORDS = set(stopwords_)

# This is getting the features using tf-idf
MIN_DF = 5  # min # fords occurence
MAX_DF = 0.5  # max occurence (percentage) in the documents
MAX_FEATURES = 1500  # most frequently occurring words

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES, min_df=MIN_DF, max_df=MAX_DF, stop_words=STOPWORDS)
tfidf = vectorizer.fit(tweets_dataset)
vectorized_tweets = vectorizer.fit_transform(tweets_dataset)
print(vectorized_tweets.shape)
vectorized_tweets = vectorized_tweets.toarray()

# classification using SVM
if USE_SVM:
    svm_model = pickle.load(open(svm_path, 'rb'))
    predict = svm_model.predict(vectorized_tweets)
    print("\nSVM Prediciton: ")

    depressed_index = np.where(predict == 1)
    depressed_index = np.asarray(depressed_index).flatten()

    print(depressed_index.size)
    print(depressed_index)

    df_depressed_tweets = np.array(tweets_dataset[depressed_index])
    pd.DataFrame(df_depressed_tweets).to_csv(
        "../predicted-tweets/predict-processed-"+file_num+".csv")

    df_depressed_tweets = np.array(tweets_text[depressed_index])
    pd.DataFrame(df_depressed_tweets).to_csv(
        "../predicted-tweets/predict-text-"+file_num+".csv")
    # score = svm_model.score(vectorized_tweets, labels)
    # print(score)


# classification using Naive Bayes
if USE_NAIVE_BAIYES:
    naive_bayes_model = pickle.load(open(naive_bayes_path, 'rb'))
    print(naive_bayes_model)
    predict = naive_bayes_model.predict(vectorized_tweets)
    
    depressed_index = np.where(predict == 1)
    depressed_index = np.asarray(depressed_index).flatten()

    print(depressed_index.size)
    print(depressed_index)

    df_depressed_tweets = np.array(tweets_dataset[depressed_index])
    pd.DataFrame(df_depressed_tweets).to_csv(
        "../predicted-tweets/predict-processed-"+file_num+".csv")

    df_depressed_tweets = np.array(tweets_text[depressed_index])
    pd.DataFrame(df_depressed_tweets).to_csv(
        "../predicted-tweets/predict-text-"+file_num+".csv")

    # score = naive_bayes_model.score(vectorized_tweets, labels)
    # print(score)


# classification using Random Forest
if USE_RANDOM_FOREST:
    random_forest__model = pickle.load(open(random_forest_path, 'rb'))
    predict = random_forest__model.predict(vectorized_tweets)
    print("\nRandom Forest Prediciton: ")
    
    depressed_index = np.where(predict == 1)
    depressed_index = np.asarray(depressed_index).flatten()

    print(depressed_index.size)
    print(depressed_index)

    df_depressed_tweets = np.array(tweets_dataset[depressed_index])
    pd.DataFrame(df_depressed_tweets).to_csv(
        "../predicted-tweets/predict-processed-"+file_num+".csv")

    df_depressed_tweets = np.array(tweets_text[depressed_index])
    pd.DataFrame(df_depressed_tweets).to_csv(
        "../predicted-tweets/predict-text-"+file_num+".csv")
    # score = random_forest__model.score(vectorized_tweets, labels)
    # print(score)
