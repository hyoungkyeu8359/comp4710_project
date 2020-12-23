# AUTHOR OF THIS CODE IS: Daniel Low, used from - https://github.com/danielmlow/reddit
# Some portion of code is written by Devin Efendy
import string
import os
import sys
import re
import pandas as pd
import numpy as np
import datetime
import stanza
import textacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Do this once at the beginning so we don't reload for each post
# nltk.download('vader_lexicon')
# nltk.download('stopwords')
# stanza.download('en')  # download English model
# nlp = stanza.Pipeline('en') # initialize English neural pipeline with entity recog NER
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
stemmer = SnowballStemmer(language='english')

"""
We don't have liwc so we don't need this

input_path_liwc = '/content/drive/My Drive/ML4HC_Final_Project/data/input/liwc_english_dictionary/' #you must obtain a LIWC license to use the dictionary
liwc_dict = np.load(input_path_liwc + 'liwc.npy', allow_pickle=True).item()
liwc_categories = list(liwc_dict.keys())
liwc_categories.sort()
"""

sid = SentimentIntensityAnalyzer()

# Count number times a specific phrase appears in the text.


def count_words(doc, phrases=[]):
    # remove punctuation except apostrophes because we need to search for things like "don't want to live"
    text = re.sub("[^\w\d'\s]+", '', doc.lower())
    counter = 0
    for phrase in phrases:
        counter += text.count(phrase)
    return counter

# Count number of times punctuation appears in the text


def punctuation_count(doc):
    d = {}
    # removed ', <>, _ because m_c_a_t doesnt reflect punctuation really
    punctuation = '!"#$%&()*+,-./:;=?@[\]^`{|}~'
    for c in doc:
        if c in punctuation:
            if c not in d:
                d[c] = 0
            d[c] += 1
    total = np.sum(list(d.values()))
    return total

# Count the number of words that belong to LIWC categories.


def liwc(liwc_path=None, document=None):
    liwc_vector = []
    # tokenize and stem
    document_tokenized = [n.strip(string.punctuation).lower()
                          for n in document.split()]
    document_stemmed = [stemmer.stem(word) for word in document_tokenized]
    for category in liwc_categories:
        counter = 0
        # for each word in category, check if its in stemmed sentence list
        counter_doc = np.sum([sum(word.rstrip() == s for s in document_stemmed)
                              for word in liwc_dict.get(category)])
        counter += counter_doc
        liwc_vector.append(counter)

    return liwc_vector


def tfidf(X_train_sentences=[], X_test_sentences=[], lower_case=True, ngram_range=(1, 2), max_features=512, min_df=2, max_df=0.8, model='vector'):
    """
    TfidfVectorizer is CountVectorizer followed by TfidfTransformer, The former converts text documents to a sparse matrix of token counts.
    This sparse matrix is then put through the TfidfTransformer which converts a count matrix to a normalized Term Frequency-Inverse Document Frquency(tf)  
    representation which is a metric of word importance. 
    We fit_transform on train_sentences and transform on test sentences to prevent overfitting, X_test_sentences can be None

    Token pattern can be included to include apostophes: token_pattern=r"\b\w[\w']+\b"

    model: {vector, sequential} depending on what model takes as input: vector (svm, random forest), sequential (lstm)
    """
    sw = stopwords.words('english')

    if model == 'sequential':
        #
        vectorizer = TfidfVectorizer(lowercase=lower_case, ngram_range=ngram_range, stop_words=sw,
                                     max_features=max_features, min_df=min_df, max_df=max_df, analyzer=lambda x: x)
        train_vectors = vectorizer.fit_transform(X_train_sentences).toarray()
        if X_test_sentences:
            test_vectors = vectorizer.transform(X_test_sentences).toarray()

    else:
        # model = 'vector'
        vectorizer = TfidfVectorizer(lowercase=lower_case, ngram_range=ngram_range, stop_words=sw,
                                     max_features=max_features, min_df=min_df, max_df=max_df)
        train_vectors = vectorizer.fit_transform(X_train_sentences).toarray()
        if X_test_sentences:
            test_vectors = vectorizer.transform(X_test_sentences).toarray()
    # train_vectors = vectorizer.fit_transform(X_train_sentences.ravel()).toarray()
    # test_vectors = vectorizer.transform(X_test_sentences.ravel()).toarray()
    feature_names = vectorizer.get_feature_names()
    feature_names = ['tfidf_'+n for n in feature_names]
    if X_test_sentences:
        return train_vectors, test_vectors, feature_names
    else:
        return train_vectors, feature_names, vectorizer


def extract_NLP_features_names(features):
    feature_names = []

    if 'readability' in features:
        feature_names.append(['automated_readability_index', 'coleman_liau_index', 'flesch_kincaid_grade_level', 'flesch_reading_ease',
                              'gulpease_index', 'gunning_fog_index', 'lix', 'smog_index', 'wiener_sachtextformel'])

    if 'basic_count' in features:
        # todo: should we be worried
        feature_names.append(['n_chars', 'n_long_words', 'n_monosyllable_words',
                              'n_polysyllable_words', 'n_sents', 'n_syllables', 'n_unique_words', 'n_words'])

    if 'sentiment' in features:
        names = ['sent_neg', 'sent_neu', 'sent_pos', 'sent_compound']
        feature_names.append(names)

    if 'covid19' in features:
        feature_names.append(['covid19_total'])

    if 'economic_stress' in features:
        feature_names.append(['economic_stress_total'])

    if 'isolation' in features:
        feature_names.append(['isolation_total'])

    if 'substance_use' in features:
        feature_names.append(['substance_use_total'])

    if 'guns' in features:
        feature_names.append(['guns_total'])

    if 'domestic_stress' in features:
        feature_names.append(['domestic_stress_total'])

    if 'suicidality' in features:
        feature_names.append(['suicidality_total'])

    if 'punctuation' in features:
        feature_names.append(['punctuation'])

    if 'liwc' in features:
        liwc_names = ['liwc_'+n for n in liwc_categories]
        feature_names.append(liwc_names)

    feature_names = [label for item in feature_names for label in item]
    return feature_names


def extract_NLP_features(doc, features):
    feature_vector = []

    if 'basic_count' or 'readability' in features:
        en = textacy.load_spacy_lang("en_core_web_sm", disable=("parser",))
        doc_ts = textacy.make_spacy_doc(doc, lang=en)
        ts = textacy.text_stats.TextStats(doc_ts)

        if 'readability' in features:
            '''
            described here: https://chartbeat-labs.github.io/textacy/build/html/api_reference/misc.html
            {'flesch_kincaid_grade_level': 15.56027397260274,
            'flesch_reading_ease': 26.84351598173518,
            'smog_index': 17.5058628484301,
            'gunning_fog_index': 20.144292237442922,
            'coleman_liau_index': 16.32928468493151,
            'automated_readability_index': 17.448173515981736,
            'lix': 65.42922374429223,
            'gulpease_index': 44.61643835616438,
            'wiener_sachtextformel': 11.857779908675797}
            '''
            try:
                dictionary = [ts.automated_readability_index,
                              ts.coleman_liau_index,
                              ts.flesch_kincaid_grade_level,
                              ts.flesch_reading_ease,
                              ts.gulpease_index,
                              ts.gunning_fog_index,
                              ts.lix,
                              ts.smog_index,
                              ts.wiener_sachtextformel]

                feature_vector.append(dictionary)
            except Exception as e:
                print(e)
                return []

        if 'basic_count' in features:
            # https://chartbeat-labs.github.io/textacy/build/html/getting_started/quickstart.html
            '''
            {'n_sents': 3,
            'n_words': 73,
            'n_chars': 414,
            'n_syllables': 134,
            'n_unique_words': 57,
            'n_long_words': 30,
            'n_monosyllable_words': 38,
            'n_polysyllable_words': 19}
            '''
            try:

                dictionary = [ts.n_chars,
                              ts.n_long_words,
                              ts.n_monosyllable_words,
                              ts.n_polysyllable_words,
                              ts.n_sents,
                              ts.n_syllables,
                              ts.n_unique_words,
                              ts.n_words]

                feature_vector.append(dictionary)
            except Exception as e:
                print("line 212: ", e)
                return []

    if 'sentiment' in features:
        # don't lowercase or remove punctuation, but maybe preprocess emojis
        scores = sid.polarity_scores(doc)
        names = ['sent_neg', 'sent_neu', 'sent_pos', 'sent_compound']
        feature_vector.append(scores.values())
        # feature_names.append(names)

    if 'covid19' in features:
        words = ['corona', 'virus', 'viral', 'covid', 'sars', 'influenza', 'pandemic', 'epidemic',
                 'quarantine', 'lockdown', 'distancing', 'national emergency', 'flatten',
                 'infect', 'ventilator', 'mask', 'symptomatic',
                 'epidemiolog', 'immun', 'incubation', 'transmission', 'vaccine', ]

        counter = count_words(doc, phrases=words)
        feature_vector.append([counter])
        # feature_names.append(['covid19_total'])

    if 'economic_stress' in features:
      # Double check that words like "rent" aren't capturing too many other words like "parent". If so replace "rent" with "pay rent" and variations
        words = ['unemploy', 'employ', 'economy', 'rent', 'mortgage', 'evict',  "enough money", "more money",
                 'pay the bills', 'owe', 'debt', "make ends meet", "pay the bills", "afford", "save enough",
                 'salary', 'wage', 'income', 'job', 'eviction']
        counter = count_words(doc, phrases=words)
        feature_vector.append([counter])
        # feature_names.append(['economic_stress_total'])

    if 'isolation' in features:
        words = ['alone', 'lonely', 'no one cares about me', 'no one cares', "can't see anyone",
                 "can't see my", "i miss my", "i want to see my", "trapped", "i'm in a cage",
                 'lonely', "feel ignored", "ignoring me" "ugly", 'rejected', "avoid", 'avoiding me', " lack of social",
                 'am single', 'been single', 'quarantine', 'lockdown', 'isolation', 'self-isolation']
        counter = count_words(doc, phrases=words)
        feature_vector.append([counter])
        # feature_names.append(['isolation_total'])

    if 'substance_use' in features:
        words = ['smoke', 'cigarette', 'tobacco', 'wine', 'drink', 'beer', 'alcohol', 'drug', 'opioid',
                 'cocaine', 'snort', 'vodka', 'whiskey', 'whisky', 'tequila', 'meth']  # (remove words like drug that may refer to when people are talking about possible covid19 drugs?)
        # we could include commonly abused prescription drugs but this would also capture healthy/prescribed use: https://www.mayoclinic.org/diseases-conditions/prescription-drug-abuse/symptoms-causes/syc-20376813
        counter = count_words(doc, phrases=words)
        feature_vector.append([counter])
        # feature_names.append(['substance_use_total'])

    if 'guns' in features:
        words = ['gun', 'pistol', 'revolver', 'semiautomatic',
                 'rifle', 'shoot', 'firearm', 'semi-automatic']
        counter = count_words(doc, phrases=words)
        feature_vector.append([counter])
        # feature_names.append(['guns_total'])

    if 'domestic_stress' in features:
        words = ['divorce', 'domestic violence', 'abuse', 'yelling', 'fighting with my',
                 "we're fighting",  'single mom', 'single dad', 'single parent', 'hit me',
                 'slapped me', 'fighting', 'fight']
        counter = count_words(doc, phrases=words)
        feature_vector.append([counter])
        # feature_names.append(['domestic_stress_total'])

    if 'suicidality' in features:
        words = ['commit suicide', 'jump off a bridge', 'I want to overdose', "i'm a burden",
                 "i'm such a burden", 'I will overdose', 'thinking about overdose', 'kill myself',
                 'killing myself', 'hang myself', 'hanging myself', 'cut myself', 'cutting myself',
                 'hurt myself', 'hurting myself', 'want to die', 'wanna die', "don't want to wake up",
                 "don't wake up", 'never want to wake up', "don't want to be alive", 'want to be alive anymore',
                 'wish it would all end', 'done with living', 'want it to end', 'it all ends tonight', 'end my life',
                 'live anymore', 'living anymore', 'life anymore', 'be dead', 'take it anymore', 'think about death',
                 'hopeless', 'hurt myself', "no one will miss me", "don't want to wake up",
                 'if I live or die', 'i hate my life', 'shoot me', 'kill me', 'suicide',
                 'no point']
        counter = count_words(doc, phrases=words)
        feature_vector.append([counter])
        # feature_names.append(['suicidality_total'])

    if 'punctuation' in features:
        count = punctuation_count(doc)
        feature_vector.append([count])
        # feature_names.append(['punctuation'])

    if 'liwc' in features:
        vector = liwc(liwc_path=None, document=doc)
        feature_vector.append(vector)
        # feature_names.append(names)

    feature_vector = [n for i in feature_vector for n in i]
    return feature_vector

csv_files = [1, 2, 3, 10, 16, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93, 100,
             107, 114, 121, 128, 135, 142, 149, 156, 163, 170, 177, 184, 191, 198, 205]

in_file_num = sys.argv[1]
print("tweets_"+in_file_num)

file_num = in_file_num
df_path = "./../cleaned-data/cleaned-data-"+file_num+".csv"
df = pd.read_csv(df_path, lineterminator='\n')

original_texts = df[pd.notnull(df['text'])]
original_texts = original_texts['text'].values

tweets = df[pd.notnull(df['tweet_processed'])]
tweets = tweets['tweet_processed'].astype(str).values
# tweets = tweets.values

features = ['sentiment',
            'isolation',
            'basic_count',
            'readability',
            'suicidality',
            'economic_stress',
            'substance_use',
            'guns',
            'domestic_stress',
            'punctuation']

feature_names = extract_NLP_features_names(features)
tweet_featres_column = ['tweet'] + feature_names
tweets_features = pd.DataFrame(columns=tweet_featres_column)
print(feature_names)
print("Features size: ", len(feature_names))

# for ti in range(len(tweets)):
for ti in range(len(tweets)):
    tweet = tweets[ti]
    feature_vector = extract_NLP_features(tweet, features)
    # print(ti, tweet)

    tweets_features_append = pd.Series([tweet] + feature_vector, index=tweet_featres_column)
    tweets_features = tweets_features.append(
        tweets_features_append, ignore_index=True)

tweets_features.to_csv('./../features/tweets_'+file_num+'_features.csv', index=False)
