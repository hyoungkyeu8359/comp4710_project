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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


#gets tweets dataframe to work with
my_path = os.path.dirname( __file__) # path of this program
df_path = my_path + "/../cleaned-data/cleaned-data-1.csv"

df = pd.read_csv(df_path)

orig_tweets = df.iloc[:, -1].values
tweets = df.iloc[:, -1].values

#preprocess data to clean it
processed_features = []
for sentence in range(0, len(tweets)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(tweets[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)


stopwords_ = stopwords.words('english')
stopwords_.extend(["im", "ive"])
STOPWORDS = set(stopwords_)

# This is getting the features using tf-idf
MIN_DF = 1 #min occurence (percentage) in the document
MAX_DF = 0.6 #max occurence (percentage) in the documents
MAX_FEATURES = 2500 #most frequently occurring words

vectorizer = TfidfVectorizer (max_features=MAX_FEATURES, min_df=MIN_DF, max_df=MAX_DF, stop_words=STOPWORDS)
tfidf = vectorizer.fit(processed_features)
processed_features = vectorizer.fit_transform(processed_features).toarray()

#load trained model
trained_model_path = my_path + '/../trained-model/random-forest-model.pkl'
with open(trained_model_path, 'rb') as file:
    text_classifier = pickle.load(file)

#making predictions
predictions = text_classifier.predict(processed_features)

orig_tweets = np.array(orig_tweets)
labels = np.array(predictions)

df = pd.DataFrame({"tweets" : orig_tweets, "labels" : labels})
df.sort_values(by=['labels'], inplace=True)
df.to_csv(my_path + "/predicted.csv", index=False)


