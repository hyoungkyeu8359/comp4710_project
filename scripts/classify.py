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

# preprocess data to clean it
processed_features = []
for sentence in range(0, len(tweets)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(tweets[sentence]))

    # remove all single characters
    processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)

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
tfidf = vectorizer.fit(processed_features)
processed_features = vectorizer.fit_transform(processed_features).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    processed_features, labels, test_size=0.9, random_state=0)

# classification using SVM
svm_model = pickle.load(open(svm_path, 'rb'))
predict = svm_model.predict(X_test[:20])
print("\nSVM Prediciton: ")
print(predict)


# classification using Naive Bayes
naive_bayes_model = pickle.load(open(naive_bayes_path, 'rb'))
predict = naive_bayes_model.predict(X_test[:20])
print("\nNaive Bayes Prediciton: ")
print(predict)

# classification using Random Forest
random_forest__model = pickle.load(open(random_forest_path, 'rb'))
print("\nRandom Forest Prediciton: ")
predict = random_forest__model.predict(X_test[:20])
print(predict)
