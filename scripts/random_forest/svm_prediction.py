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

svm_path = 'trained_model\\svm-model.sav'



csv_files = [1, 2, 3, 10, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93, 100,
             107, 114, 121, 128, 135, 142, 149, 156, 163, 170, 177, 184, 191, 198, 205]

max_abs_scaler = preprocessing.MaxAbsScaler()
svm_model = pickle.load(open(svm_path, 'rb'))

START_INDEX = 3
END_INDEX = 286
for file in csv_files:
    file_name = str(file)
    print("File Number: " + file_name)

    df = pd.read_csv("data\\input\\feature_extraction\\tfidf_combined\\cleaned-data-"+file_name+ "_features_tfidf_256"+".csv", lineterminator="\n")
    print(df.shape)
    df['post'].replace('',np.nan,inplace=True)
    df.dropna(subset=['post'],inplace=True)
    print(df.shape)
    
    df['original_text'].replace('',np.nan,inplace=True)
    df.dropna(subset=['original_text'],inplace=True)
    print(df.shape)

    df = df.drop_duplicates(subset=['original_text', 'post'], keep=False)
    print(df.shape)
    original_text = df['original_text'].astype(str).values
    post = df['post'].astype(str).values


    prediction_features = df.iloc[:, START_INDEX:END_INDEX].values
    print("Shape for prediction_features: " + str(prediction_features.shape))
    print("Features: " + str(prediction_features))
    print("Start normalizing data")
    X_predictions_maxabs = max_abs_scaler.fit_transform(prediction_features)
    print("Shape for X_predictions_maxabs: " + str(X_predictions_maxabs.shape))
    predict = svm_model.predict(X_predictions_maxabs)



    print("\nSVM Prediciton: ")
    result_df = df[['original_text','author','post']]
    result_df.insert(loc=3, column='label', value=predict)
    print(result_df['label'].value_counts())

    result_df.to_csv("output\\prediction-"+file_name+".csv", index=None)
    


    
