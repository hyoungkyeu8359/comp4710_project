import numpy as np
import pandas as pd
import re
import sys
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import os.path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import re
from sklearn.model_selection import train_test_split
from sklearn import svm, preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

USE_SVM = True
USE_NAIVE_BAIYES = False
USE_RANDOM_FOREST = False

svm_path = './../trained-model/svm-model.sav'
naive_bayes_path = './../trained-model/naive-baiyes-model.sav'
random_forest_path = './../trained-model/random-forest-model.sav'

svm_vectorizer_path = './../trained-model/vectorizer/svm-vectorizer.sav'
naive_bayes_vectorizer_path = './../trained-model/vectorizer/naive-baiyes-vectorizer.sav'
# random_forest_vectorizer_path = './../trained-model/vectorizer/random-forest-vectorizer.sav'

csv_files = [1, 2, 3, 10, 16, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93, 100,
             107, 114, 121, 128, 135, 142, 149, 156, 163, 170, 177, 184, 191, 198, 205]

in_file_num = sys.argv[1]
print("Classifying tweets_"+in_file_num)

file_num = in_file_num
df_path = "./../cleaned-data/cleaned-data-"+file_num+".csv"
df = pd.read_csv(df_path, lineterminator='\n')
# original_texts = df['text'].values
original_texts = df[pd.notnull(df['text'])]
original_texts = original_texts['text'].astype(str).values

# tweets = df['tweet_processed'].values
tweets_df = df[pd.notnull(df['tweet_processed'])]
tweets = tweets_df['tweet_processed'].astype(str).values
original_texts = tweets_df['text'].astype(str).values
hashtags = tweets_df['hashtags']
hashtags = hashtags.replace(np.nan, '', regex=True)
hashtags = hashtags.astype(str).values

df_tweet_feature = pd.read_csv(
    "./../features/tweets_"+file_num+"_features.csv", lineterminator='\n')
test_features = df_tweet_feature.iloc[:, 1:29].values

# preprocess data to clean it
tweets_dataset = []
tweets_dataset_index = []
tweets_text = []
tweets_hashtags = []

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

    if tweets[sentence] not in tweets_dataset:
        tweets_dataset.append(processed_tweet)
        tweets_dataset_index.append(sentence)
        tweets_text.append(original_texts[sentence])
        tweets_hashtags.append(hashtags[sentence])

tweets_dataset = np.array(tweets_dataset)
tweets_text = np.array(tweets_text)
tweets_hashtags = np.array(tweets_hashtags)

# print("tweets_dataset: ", tweets_dataset[0])
# print("tweets_text: ", tweets_text[0])
# print("tweets_hashtags: ", tweets_hashtags[0])

# exit()

# classification using SVM
if USE_SVM:
    svm_model = pickle.load(open(svm_path, 'rb'))
    svm_vectorizer = pickle.load(open(svm_vectorizer_path, 'rb'))
    vectorized_tweets = svm_vectorizer.transform(tweets_dataset).toarray()

    print(test_features.shape)
    test_features = test_features[tweets_dataset_index]
    # print(test_features.shape)

    test_all_features = np.hstack((test_features, vectorized_tweets))

    max_abs_scaler = preprocessing.MaxAbsScaler()
    X_test_maxabs = max_abs_scaler.fit_transform(test_all_features)

    print(X_test_maxabs.shape)

    result_column_names = ["original_tweet", "processed_tweet", "hashtags"]

    predict = svm_model.predict(X_test_maxabs)
    print("\nSVM Prediciton on: " + "cleaned-data-"+file_num+".csv")

    depressed_index = np.where(predict == 1)
    depressed_index = np.asarray(depressed_index).flatten()

    print("total_unique_tweets: ", tweets_dataset.size)
    print("total_negative_tweets: ", depressed_index.size)
    print("total_positive_tweets: ", tweets_dataset.size - depressed_index.size)

    result_df = pd.DataFrame(columns=result_column_names)
    original_tweet = tweets_text[depressed_index]
    processed_tweet = tweets_dataset[depressed_index]
    hashtags_result = tweets_hashtags[depressed_index]

    data = {'original_tweet': original_tweet, 
        'processed_tweet': processed_tweet, 
        'hashtags': hashtags_result} 

    result_df = pd.DataFrame(data)

    result_df.to_csv(
        "../predicted-tweets/prediction-"+file_num+".csv", index=None)

    # df_depressed_tweets = np.array(tweets_dataset[depressed_index])
    # pd.DataFrame(df_depressed_tweets).to_csv(
    #     "../predicted-tweets/predict-processed-"+file_num+".csv")

    # df_depressed_tweets = np.array(tweets_text[depressed_index])
    # pd.DataFrame(df_depressed_tweets).to_csv(
    #     "../predicted-tweets/predict-text-"+file_num+".csv")


# classification using Naive Bayes
if USE_NAIVE_BAIYES:
    naive_bayes_model = pickle.load(open(naive_bayes_path, 'rb'))
    naive_bayes_vectorizer = pickle.load(
        open(naive_bayes_vectorizer_path, 'rb'))
    vectorized_tweets = naive_bayes_vectorizer.transform(
        tweets_dataset).toarray()

    predict = naive_bayes_model.predict(vectorized_tweets)
    print("\nNaive Bayes Prediciton: ")

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
