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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import pickle

#gets tweets dataframe to work with
my_path = os.path.dirname( __file__) # path of this program
# df_path = my_path + "./../swcwang-final-dataset/tweets_combined_labeled1.csv"

df_path = "./../swcwang-final-dataset/tweets_combined_labeled1.csv"
df = pd.read_csv(df_path)

tweets = df.iloc[:, 0].values
labels = df.iloc[:, 1].values


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
MIN_DF = 10 #min # fords occurence
MAX_DF = 0.8 #max occurence (percentage) in the documents
MAX_FEATURES = 2500 #most frequently occurring words

vectorizer = TfidfVectorizer (max_features=MAX_FEATURES, min_df=MIN_DF, max_df=MAX_DF, stop_words=STOPWORDS)
tfidf = vectorizer.fit(processed_features)
processed_features = vectorizer.fit_transform(processed_features).toarray()


#dividing Data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)


#training the model
text_classifier = MultinomialNB()
text_classifier.fit(X_train, y_train)

#save trained model
trained_model_path = './../trained-model/naive-baiyes-model.sav'
pickle.dump(text_classifier, open(trained_model_path, 'wb'))
# vectorizer_path = './../trained-model/vectorizer/naive-baiyes-vectorizer.sav'
# pickle.dump(tfidf, open(vectorizer_path, 'wb'))


#making predictions and evaluating the model
predictions = text_classifier.predict(X_test)


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))