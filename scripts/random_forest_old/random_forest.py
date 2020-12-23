import numpy as np 
import pandas as pd 
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

my_path = os.path.dirname( __file__) # path of this program

#gets a dataframe to work with
#use only after the csv file was processed by pre_processed.py
df_path = my_path + "/processed_reddit_twitter_combined.csv" #enter a directory after the path of this program

df = pd.read_csv(df_path)

posts = df.iloc[:, 0].values
labels = df.iloc[:, 1].values

processed_features = []
for sentence in range(0, len(posts)):
    processed_feature = str(posts[sentence])
    processed_features.append(processed_feature)

#Extending stopwords
stopwords_ = stopwords.words('english')
stopwords_.extend(["im", "ive", "", "ve", "ll", "re"])
STOPWORDS = set(stopwords_)

# This is getting the features using tf-idf
MIN_DF = 2 #min occurence (percentage) in the document
MAX_DF = 0.7 #max occurence (percentage) in the documents
MAX_FEATURES = 256 #most frequently occurring words

vectorizer = TfidfVectorizer (max_features=MAX_FEATURES, min_df=MIN_DF, max_df=MAX_DF, stop_words=STOPWORDS)
tfidf = vectorizer.fit(processed_features)
processed_features = vectorizer.fit_transform(processed_features).toarray()


#dividing Data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)


#training the model
text_classifier = RandomForestClassifier(n_estimators=500, random_state=1)
text_classifier.fit(X_train, y_train)

#save trained model
trained_model_path = my_path + '/random-forest-model.pkl'
with open(trained_model_path, 'wb') as file:
    pickle.dump(text_classifier, file)


#making predictions and evaluating the model
predictions = text_classifier.predict(X_test)


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print("Prediction score: " + accuracy_score(y_test, predictions))


