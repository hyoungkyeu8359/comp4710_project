#For the classification we use an algorithm from scikit-learn
#https://scikit-learn.org/stable/modules/naive_bayes.html 

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
import time

my_path = os.path.dirname( __file__) # path of this program
rf_path = my_path + './random-forest-model.sav'

csv_files = [1, 2, 3, 10, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93, 100,
             107, 114, 121, 128, 135, 142, 149, 156, 163, 170, 177, 184, 191, 198, 205]

max_abs_scaler = preprocessing.MaxAbsScaler()
rf_model = pickle.load(open(rf_path, 'rb'))

START_INDEX = 3
END_INDEX = 286
NUM_ITER = 10
counter = 0

rows_count = []
times = []

df = pd.read_csv(my_path + "./cleaned-data/cleaned-data-1_features_tfidf_256"+".csv", lineterminator="\n")

for file in csv_files:
    file_name = str(file)

    if(counter != 0):
        temp = pd.read_csv(my_path + "./cleaned-data/cleaned-data-"+file_name+ "_features_tfidf_256"+".csv", lineterminator="\n")
        df = df.append(temp, ignore_index = True) 

    df['post'].replace('',np.nan,inplace=True)
    df.dropna(subset=['post'],inplace=True)
    
    df['original_text'].replace('',np.nan,inplace=True)
    df.dropna(subset=['original_text'],inplace=True)

    df = df.drop_duplicates(subset=['original_text', 'post'], keep=False)
    original_text = df['original_text'].astype(str).values
    post = df['post'].astype(str).values

    prediction_features = df.iloc[:, START_INDEX:END_INDEX].values
    X_predictions_maxabs = max_abs_scaler.fit_transform(prediction_features)

    start_time = time.time()
    predict = rf_model.predict(X_predictions_maxabs)
    finish_time = time.time()

    elapsed = finish_time - start_time

    rows_count.append(len(df.index))
    times.append(elapsed)

    if (counter >= (NUM_ITER - 1)):
        break

    counter = counter + 1


df = pd.DataFrame(data={"rows count": rows_count, "time": times})
df.to_csv("./class_perf.csv", index=False)

