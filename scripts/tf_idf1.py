import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import re
from nltk.corpus import stopwords

import tf_idf_helper as TfidfHelper

path = "swcwang-final-dataset\\tweets_combined_labeled.csv"
# path = "../swcwang-final-dataset/tweets_combined.csv"
df = pd.read_csv(
    path)

tweets = df['tweet_processed'].values.astype('U')
# tweets = tweets.to_list()
# print(tweets)
# exit()

# This is getting the features using word count 
cv = CountVectorizer(max_df=0.85, max_features=10000)
word_count_vector = cv.fit_transform(tweets)

# print(list(cv.vocabulary_.keys())[:30])

def get_tfidf():
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)


    # you only needs to do this once, this is a mapping of index to
    feature_names = cv.get_feature_names()

    stopwords_ = stopwords.words('english')
    stopwords_.extend(["im", "ive"])
    STOPWORDS = set(stopwords_)

    # This is getting the features using tf-idf
    tfidf_vectorizer = TfidfVectorizer(
        min_df=10, use_idf=True, max_features=50, stop_words=STOPWORDS)

    # just send in all your docs here
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(tweets)

    feature_names = tfidf_vectorizer.get_feature_names()
    # print(feature_names)

    # print(tfidf_vectorizer_vectors.nonzero()[1])

    # for col in tfidf_vectorizer_vectors.nonzero()[1]:
    #     print (feature_names[col], ' - ', tfidf_vectorizer_vectors[0, col])
    return tfidf_vectorizer

# exit()

# '''
# This section is appending all tweets and put it in the transformer
# tweet = ''
# for i in tweets:
#     tweet = tweet + i
# '''

# tweet = tweets[0]

# # generate tf-idf for the given document
# tf_idf_vector = tfidf_transformer.transform(cv.transform([tweet]))

# # sort the tf-idf vectors by descending order of scores
# sorted_items = TfidfHelper.sort_coo(tf_idf_vector.tocoo())

# # extract only the top n; n here is 10
# keywords = TfidfHelper.extract_topn_from_vector(
#     feature_names, sorted_items, 20)

# # now print the results
# print("\n=====Doc=====")
# print(tweet)
# print("\n===Keywords===")
# for k in keywords:
#     print(k, keywords[k])

