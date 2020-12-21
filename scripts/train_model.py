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
from sklearn import svm, preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

"""
These are indeces for the columns in training dataset features.
"""
START_INDEX = 4
END_INDEX = 349
LABEL_INDEX = 350

READABILITY_START = 4  # 4 - 12
READABILITY_END = 12   # 4 - 12

COUNT_START = 13  # 13 - 20
COUNT_END = 20    # 13 - 20

SENT_START = 21  # 21 - 24
SENT_END = 24    # 21 - 24

TOPIC_START = 25  # 25 - 31
TOPIC_END = 31    # 25 - 31

LIWC_START = 32
LIWC_END = 93

TFIDF_START = 94
TFIDF_END = 349

df_training_path = "./../zenodo-dataset/final_combined.csv"
df_training_data = pd.read_csv(df_training_path, lineterminator='\n')
training_posts = df_training_data['post'].values
training_dataset = training_posts
training_dataset = [post.replace('\n', ' ').replace(
    '  ', ' ').replace('“', '').replace('”', '') for post in training_dataset]
training_labels = df_training_data.iloc[:, 350].values

training_features_1 = df_training_data.iloc[:, START_INDEX:TOPIC_END+1].values
training_features_2 = df_training_data.iloc[:, TFIDF_START:TFIDF_END+1].values
training_features = np.hstack((training_features_1, training_features_2))
print(training_features_2.shape)
# exit()

"""
This is for extracting the top n tfidf words from the training dataset
"""
# training_column_names = df_training_data.columns.values
# tfidf_words = []

# for col in training_column_names:
#     if "tfidf_" in col:
#         tfidf_words.append(col)

# print(tfidf_words)
# exit()

file_num = str(1)
df_path = "./../cleaned-data/cleaned-data-"+file_num+".csv"
df = pd.read_csv(df_path, lineterminator='\n')

original_texts = df['text'].values
tweets = df['tweet_processed'].values

# ================== Classification Code start here ==================
stopwords_ = stopwords.words('english')
stopwords_.extend(["im", "ive"])
STOPWORDS = set(stopwords_)

min_df = 2
max_df = 0.8

print("Vectorizing training dataset")
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words=STOPWORDS,
                             max_features=256, min_df=min_df, max_df=max_df)
train_tfidf = vectorizer.fit(training_dataset)
train_vector = vectorizer.fit_transform(training_dataset)
train_vector = np.array(train_vector.toarray())

print("Training features:")
print(vectorizer.get_feature_names())

training_features_1 = df_training_data.iloc[:, START_INDEX:TOPIC_END+1].values
training_features = np.hstack((training_features_1, train_vector))
print("Training features shape: ", training_features.shape)
exit()

X_train, X_test, y_train, y_test = train_test_split(training_features, training_labels, test_size=0.2)

print("Start normalizing data")
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)

print("Start training SVM")
text_classifier = svm.SVC(kernel='linear', probability=True)
text_classifier.fit(X_train_maxabs, y_train)
print("Finished training SVM")

max_abs_scaler = preprocessing.MaxAbsScaler()
X_test_maxabs = max_abs_scaler.fit_transform(X_test)
predictions = text_classifier.predict(X_test_maxabs)

#save trained model
trained_model_path = './../trained-model/svm-model.sav'
pickle.dump(text_classifier, open(trained_model_path, 'wb'))
vectorizer_path = './../trained-model/vectorizer/svm-vectorizer.sav'
pickle.dump(train_tfidf, open(vectorizer_path, 'wb'))
print("Saved trained SVM Model")

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))