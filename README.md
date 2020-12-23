# COMP4710 Group 15

# Contributors Devin Efendy, Kevin Kim, Alejo Vallega, Victor Sharonov and Joe Smith 

# Source / Credits

Dataset for training dataset:

Low, D. M., Rumker, L., Talker, T., Torous, J., Cecchi, G., & Ghosh, S. S. Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit during COVID-19: An Observational Study. Journal of medical Internet research. doi: 10.2196/22635

File used for feature extractions:
https://github.com/danielmlow/reddit/blob/master/reddit_feature_extraction.ipynb


Files used for preprocessing slang words and acronyms for our dataset of tweets:
https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt
https://github.com/rishabhverma17/sms_slang_translator/blob/master/Script.py


# Explanation of Folders

SVM_RF-results includes prediction results for Random Forest and SVM classifiers

cleaned-data includes our preprocessed tweets for our tweets dataset

features has our tweets preprocessed with included features for each tweets

final_combine_dataset_prediction is prediction results using SVM for a different dataset in zenodo-dataset

result is graphs for the prediction results using SVM for a different dataset in zenodo-dataset

scripts contains various python scripts for training, classifying, dataset modifications and others

trained-model includes our saved SVM model and its vectorizer for a different dataset in zenodo-dataset

zenodo-dataset contains our original and firstdataset used for SVM that differs from the updated Random Forest and SVM dataset 

# Explanation of Scripts

random_forest_final and svm_final contain training and classification predictions for svm and random forest on a dataset named all_combined 

classify.py is used for classification for SVM on a dataset named final_combined

feature_extraction.py is a feature extraction script from Daniel Low 

preprocess_data.py is used to preprocess our tweets

random_sample.py is to reduce each of our csv tweets to 20 000 tweets

remove_cols.py removes unnecessary columns for our tweet dataset

slang.txt and slang_script.py is used for preprocessing slang and acronyms

tf_idf.py and tf_idf_helper.py is for feature extraction

train_model.py is a training model for SVM for final_comvbined.csv

tweet_wordcloud.py is a graph to represent the prediction results from the trained model we get from train_model.py





