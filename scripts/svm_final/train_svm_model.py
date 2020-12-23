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
import time

my_path = os.path.dirname( __file__) # path of this program

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

df_training_path = my_path + "./all_combined.csv"
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


print("Start training SVM")
start_time = time.time()
text_classifier = svm.SVC(kernel='linear', probability=True)
text_classifier.fit(X_train_maxabs, y_train)
finish_time = time.time()
print("Finished training SVM")

max_abs_scaler = preprocessing.MaxAbsScaler()
X_test_maxabs = max_abs_scaler.fit_transform(X_test)
predictions = text_classifier.predict(X_test_maxabs)

#save trained model
trained_model_path = my_path + './svm-model.sav'
pickle.dump(text_classifier, open(trained_model_path, 'wb'))
print("Saved trained SVM Model")

print('Elapsed time to train the model: %f\n' % (finish_time - start_time))
print('Confusion matrix:\n')
print(confusion_matrix(y_test,predictions))
print('\nClassification report:\n')
print(classification_report(y_test,predictions))
print("Prediction score: %f\n" % (accuracy_score(y_test, predictions)).item())

with open(my_path + './train_performance.txt', 'w') as my_file:
    my_file.write('Elapsed time to train the model (in seconds): %f\n' % (finish_time - start_time))
    my_file.write('Confusion matrix:\n')
    my_file.write(np.array2string(confusion_matrix(y_test,predictions)))
    my_file.write('\nClassification report:\n')
    my_file.write(classification_report(y_test,predictions))
    my_file.write("Prediction score: %f\n" % (accuracy_score(y_test, predictions)).item())

