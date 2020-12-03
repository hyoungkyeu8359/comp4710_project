from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
from tf_idf1 import get_tfidf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


data = pd.read_csv("swcwang-final-dataset\\tweets_combined_labeled.csv")
X_train, X_test, y_train, y_test = train_test_split(data['tweet_processed'], data['label'], test_size = 0.2, random_state = 1)
X_train = X_train.values.astype('U')
X_test = X_test.values.astype('U')
y_train = y_train.values.astype('U')
y_test = y_test.values.astype('U')

vectorizer = get_tfidf()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf =  vectorizer.fit_transform(X_test)

svm = svm.SVC(C=1000, kernel='rbf', degree=3, gamma='auto')
svm.fit(X_train_tfidf, y_train)


# predict the labels on validation dataset
predictions_svm = svm.predict(X_test_tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_svm, y_test)*100)