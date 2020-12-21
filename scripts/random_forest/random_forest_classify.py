import numpy as np 
import pandas as pd 
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import os.path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


#gets tweets dataframe to work with
#use only after the csv file was processed by pre_processed.py
my_path = os.path.dirname( __file__) # path of this program
df_path = my_path + "/../../cleaned-data/cleaned-data-1.csv"

df = pd.read_csv(df_path)

orig_tweets = df.iloc[:, -1].values #untouched tweets
tweets = df.iloc[:, -1].values

processed_features = []
for sentence in range(0, len(tweets)):
    processed_feature = str(tweets[sentence])
    processed_features.append(processed_feature)


#Extending stopwords
stopwords_ = stopwords.words('english')
stopwords_.extend(["im", "ive", "", "ve", "ll", "re"])
STOPWORDS = set(stopwords_)


# This is getting the features using tf-idf
MIN_DF = 3 #min occurence (percentage) in the document
MAX_DF = 0.8 #max occurence (percentage) in the documents
MAX_FEATURES = 500 #most frequently occurring words

vectorizer = TfidfVectorizer (max_features=MAX_FEATURES, min_df=MIN_DF, max_df=MAX_DF, stop_words=STOPWORDS)
tfidf = vectorizer.fit(processed_features)
processed_features = vectorizer.fit_transform(processed_features).toarray()

#load trained model
trained_model_path = my_path + '/random-forest-model.pkl'
with open(trained_model_path, 'rb') as file:
    text_classifier = pickle.load(file)

#making predictions
predictions = text_classifier.predict(processed_features)

orig_tweets = np.array(orig_tweets)
labels = np.array(predictions)

#write results to a csv file
df = pd.DataFrame({"tweets" : orig_tweets, "labels" : labels})
df.sort_values(by=['labels'], inplace=True)
df.to_csv(my_path + "/predicted.csv", index=False)


