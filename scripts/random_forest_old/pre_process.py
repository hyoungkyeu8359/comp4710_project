import numpy as np 
import pandas as pd 
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import os.path

my_path = os.path.dirname( __file__) # path of this program

#gets a dataframe to work with
df_path = my_path + "/reddit_twitter_combined.csv" #enter a directory after the path of this program

df = pd.read_csv(df_path)

posts = df.iloc[:, 0].values
labels = df.iloc[:, 1].values #uncomment if the dataset is for training, comment if the dataset for prediction


#preprocess data to clean it, tokenize, stem and lemmatize
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
processed_features = []
for sentence in range(0, len(posts)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(posts[sentence]))

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    # remove numbers
    processed_feature= re.sub(r'\d+', '', processed_feature)

    # remove all single characters
    processed_feature= re.sub(r'\b[a-zA-Z]\b', ' ', processed_feature)

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature)

    # remove starting space
    processed_feature= re.sub(r'^\s+', '', processed_feature)

    # remove ending space
    processed_feature= re.sub(r'\s+$', '', processed_feature)

    # remove prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    #Tokenization
    tokens = word_tokenize(processed_feature)
    
    #print(tokens)
    #Stemming
    stemmed_tokens = []
    for word in tokens:
        stemmed_tokens.append(stemmer.stem(word))
    

    #Lemmatization
    lemmatized_tokens = []
    for word in stemmed_tokens:
        lemmatized_tokens.append(lemmatizer.lemmatize(word))

    processed_feature = ' '.join(lemmatized_tokens)
    processed_features.append(processed_feature)


#write processed dataset to csv file
#processed_df = pd.DataFrame({'post': processed_features}) #uncomment if the dataset for predicting
processed_df = pd.DataFrame({'post': processed_features, 'label': labels}) #uncomment if the dataset for training
processed_df.to_csv(my_path + '/processed_reddit_twitter_combined.csv', index=False) #name your csv file