import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import re
from nltk.corpus import stopwords

import tf_idf_helper as TfidfHelper

path = "../swcwang-final-dataset/tweets_combined_target_1.csv"
df = pd.read_csv(path)

tweets = df['tweet_processed'].values.astype('U')

# This is getting the features using word count
cv = CountVectorizer(max_df=0.85, max_features=10000)
word_count_vector = cv.fit_transform(tweets)

# print(list(cv.vocabulary_.keys())[:30])

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)


# # you only needs to do this once, this is a mapping of index to
features = cv.get_feature_names()

stopwords_ = stopwords.words('english')
stopwords_.extend(["im", "ive"])
STOPWORDS = set(stopwords_)

# This is getting the features using tf-idf
MIN_DF = 10
MAX_FEATURES = 1000
TOP_N = 50
tfidf_vectorizer = TfidfVectorizer(
    min_df=MIN_DF, use_idf=True, max_features=MAX_FEATURES, stop_words=STOPWORDS)

# just send in all your docs here
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(tweets)
Xtr = tfidf_vectorizer_vectors

features = tfidf_vectorizer.get_feature_names()

"""
Let’s see if this topic is represented also in the overall corpus. 
For this, we will calculate the average tf-idf score of all words across a number of
documents (in this case all documents), i.e. the average per column of a tf-idf matrix:
"""

print("the average per column of a tf-idf matrix:")
print(TfidfHelper.top_mean_feats(Xtr, features,
                                 grp_ids=None, min_tfidf=0.1, top_n=TOP_N))

exit()

'''
This section is appending all tweets and put it in the transformer
tweet = ''

for i in tweets:
    tweet = tweet + i

'''

tweet = tweets[10]

# generate tf-idf for the given document
tf_idf_vector = tfidf_transformer.transform(cv.transform([tweet]))

# sort the tf-idf vectors by descending order of scores
sorted_items = TfidfHelper.sort_coo(tf_idf_vector.tocoo())

# extract only the top n; n here is 10
keywords = TfidfHelper.extract_topn_from_vector(
    features, sorted_items, 20)

# now print the results
print("\n=====Doc=====")
print(tweet)
print("\n===Keywords===")
for k in keywords:
    print(k, keywords[k])
