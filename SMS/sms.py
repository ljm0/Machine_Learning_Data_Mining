import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, svm
from sklearn.model_selection import (train_test_split, learning_curve, StratifiedShuffleSplit, GridSearchCV, cross_val_score)
df = pd.read_csv('data/SMSCollection.csv', sep="m;")
# sms = pd.read_csv('SmsCollectionTabDelimited.csv', '\t', header=0)
df['label'].replace('ha','ham',inplace=True)
df['label'].replace('spa','spam',inplace=True)
y = df['label']
le = LabelEncoder()
y_enc = le.fit_transform(y)

processed = raw_text.str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b',
                                 'emailaddr')
processed = processed.str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',
                                  'httpaddr')
processed = processed.str.replace(r'£|\$', 'moneysymb')
processed = processed.str.replace(
    r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
    'phonenumbr')
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
# print(df.head())
# print(df['label'].value_counts())
