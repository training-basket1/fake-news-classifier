##########################
###   Text DATA PREP   ###
##########################

from collections import Counter

from tensorflow import keras
from keras import preprocessing


import nltk
# nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

def process_labels(labels):
    if labels.dtype == 'object':
        lbl_enc = preprocessing.LabelEncoder()
        labels = lbl_enc.fit_transform(labels)
    return labels

def convert_categorical_features(df,cat_columns):
    for cat in cat_columns:
        df[cat] = keras.utils.to_categorical(df[cat], num_classes=2)

## Cleaning text from unused characters
def clean_text(text):
    text = str(text).replace(r'http[\w:/\.]+', ' ')  # removing urls
    text = str(text).replace(r'[^\.\w\s]', ' ')  # remove everything but characters and punctuation
    text = str(text).replace('[^a-zA-Z]', ' ')
    text = str(text).replace(r'\s\s+', ' ')
    text = text.lower().strip()
    #text = ' '.join(text)
    return text

## Nltk Preprocessing include:
# Stop words,
# Stemming and
# For our project we use only Stop word removal
def nltk_preprocesing(text):
    text = ' '.join([word for word in text.split() if word not in stopwords_dict])
    #text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    return  text

## Merge Text features together
def merge_text_features(df, text_features , col_name = 'news'):
    print(" Features in dataset :: ",df.columns)
    df[col_name]=df[text_features].agg(' '.join, axis=1)
    print("Merge news text statistics::\n ",df.news.str.split().str.len().describe())
    return df

## Preperaing Datasets
def preparing_datasets(df, text_features):
    XY = df.copy()
    XY = merge_text_features(XY, text_features)
    XY["news"] = XY.news.apply(clean_text)    
    print(" Cleaning as remove special character is done..")
    # print(XY.head())
    XY["news"] = XY.news.apply(nltk_preprocesing)
    X = XY['news']
    y = XY.label
    print("Text len statistic after Merge news and preprocessing::\n ",X.str.split().str.len().describe())
    if  y.dtype=='object':
        y= process_labels(y)
    return X,y