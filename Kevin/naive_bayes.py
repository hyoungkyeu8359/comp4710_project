from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB # Multinomial Naive Bayes Model
from sklearn.metrics import accuracy_score # For calculating accuracy of the model

newsdata=fetch_20newsgroups(subset='train')

# Check if the data has been imported successfully

print(newsdata.keys())
print (len(newsdata.data), len(newsdata.filenames), len(newsdata.target_names), len(newsdata.target))
print(newsdata.target_names)
print(newsdata.target[0])
print(newsdata.target_names[7])
print(newsdata.data[0])

# BoW (Bag of Words)
dtmvector = CountVectorizer()
X_train_dtm = dtmvector.fit_transform(newsdata.data)
print(X_train_dtm.shape)

# TF-IDF
tfidf_transformer = TfidfTransformer()
tfidfv = tfidf_transformer.fit_transform(X_train_dtm)
print(tfidfv.shape)

# Multinomial Naive Bayes Model
mod = MultinomialNB()
mod.fit(tfidfv, newsdata.target)

# 
newsdata_test = fetch_20newsgroups(subset='test', shuffle=True) # Loads test data
X_test_dtm = dtmvector.transform(newsdata_test.data) # Test data into DTM
tfidfv_test = tfidf_transformer.transform(X_test_dtm) # DTM into TF-IDF

predicted = mod.predict(tfidfv_test) # Prediction on Test data
print("Accuracy:", accuracy_score(newsdata_test.target, predicted)) # Compare prediction and actual data