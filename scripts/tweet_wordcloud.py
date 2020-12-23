import numpy as np
import pandas as pd
import pickle
import os.path
import sys
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import re
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

stopwords_ = stopwords.words('english')
stopwords_.extend(["im", "ive"])
STOPWORDS = set(stopwords_)

min_df = 2
max_df = 0.8

df_training_path = "./../zenodo-dataset/final_combined.csv"
df_training_data = pd.read_csv(df_training_path, lineterminator='\n')
training_posts = df_training_data['post'].values
training_dataset = training_posts
training_dataset = [post.replace('\n', ' ').replace(
    '  ', ' ').replace('“', '').replace('”', '') for post in training_dataset]
training_dataset = [post.lower() for post in training_dataset]

predicted_dir = "./../predicted-tweets/"

csv_files = [1, 2, 3, 10, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93, 100,
             107, 114, 121, 128, 135, 142, 149, 156, 163, 170, 177, 184, 191, 198, 205]

predicted_tweets = []

for file in csv_files:
    file_num = str(file)
    predicted_path = predicted_dir + "prediction-" + file_num + ".csv"
    df_predicted = pd.read_csv(predicted_path, lineterminator='\n')
    tweets = df_predicted['processed_tweet'].values
    predicted_tweets.extend(tweets)
    print("Size: ", len(predicted_tweets))

# svm_vectorizer_path = './../trained-model/vectorizer/svm-vectorizer.sav'
# vectorizer = pickle.load(open(svm_vectorizer_path, 'rb'))
# tfidf_predicted = vectorizer.transform(predicted_tweets)

vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words=STOPWORDS,
                             max_features=256, min_df=min_df, max_df=max_df)
train_vector = vectorizer.fit_transform(training_dataset)
d = vectorizer.vocabulary_

d.pop("years")
d.pop("x200b")
d.pop("amp")
d.pop("amp x200b")

# Create and generate a word cloud image:
wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)

# Display the generated image:
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

exit()

# start of training data

print("Vectorizing training dataset")
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words=STOPWORDS,
                             max_features=256, min_df=min_df, max_df=max_df)
train_tfidf = vectorizer.fit(training_dataset)
train_vector = vectorizer.fit_transform(training_dataset)
d = vectorizer.vocabulary_

d.pop("years")
d.pop("x200b")
d.pop("amp")
d.pop("amp x200b")

# Create and generate a word cloud image:
wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)

# Display the generated image:
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
