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
from sklearn.ensemble import RandomForestClassifier
"""
These are indeces for the columns in training dataset features.
"""
START_INDEX = 4
END_INDEX = 287
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

df_training_path = "E:\\COMP4710_project_last\\data\\combined\\all_combined.csv"
df_training_data = pd.read_csv(df_training_path, lineterminator='\n')

training_labels = df_training_data.iloc[:, 288].values

training_features_1 = df_training_data.iloc[:, START_INDEX:END_INDEX].values

print("Shape for prediction_features: " + str(training_features_1.shape))
print("Features: " + str(training_features_1))






X_train, X_test, y_train, y_test = train_test_split(training_features_1, training_labels, test_size=0.2)

print("Start normalizing data")
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
print("Shape for X_predictions_maxabs: " + str(X_train_maxabs.shape))


print("Start training Random Forest")
text_classifier = RandomForestClassifier(n_estimators=500, random_state=1)
text_classifier.fit(X_train_maxabs, y_train)
print("Finished training Random Forest")

max_abs_scaler = preprocessing.MaxAbsScaler()
X_test_maxabs = max_abs_scaler.fit_transform(X_test)
predictions = text_classifier.predict(X_test_maxabs)

#save trained model
trained_model_path = 'E:\\COMP4710_project_last\\trained_model\\svm-model.sav'
pickle.dump(text_classifier, open(trained_model_path, 'wb'))
print("Saved trained SVM Model")

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

