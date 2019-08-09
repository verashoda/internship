#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:13:56 2019
Unsupervised clustering with Kmeans (Version 1) resulted with only 1 sample?
@author: verareyes
"""

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
import sklearn
from sklearn import feature_extraction
import mpld3
import string

##### Loading the data ######
df = pd.read_csv('/home/verareyes/lp/mine_01.txt', sep='[<>]', engine='python', error_bad_lines=False, names=('time','user','message'))
df1 = pd.DataFrame(df)
#time = df['time']
#time_list = df['time'].to_list()
#message = df['message']
#message_list = df['message'].to_list()

##### Stopwords ######
defined_stop_words = ['pogchamp', 'lul', 'biblethump', 'babyrage', 'kappa', 'kreygasm']
j = list(string.punctuation) + defined_stop_words
stopwords = j

def preprocess(x):
    x = re.sub('[^a-z\s]','',x.lower()) #remove noise
    x = [w for w in x.split() if w not in set(stopwords)] #remove stopwords
    return ' '.join(x) #join the list

df1['message'] = df1['message'].apply(preprocess)

####### word tokenize ######
df1['words'] = df1.message.str.strip().str.split('[\W_]+')

##### Flatten Data #####
rows = list()
for row in df1[['time', 'words']].iterrows():
    r = row[1]
    for word in r.words:
        rows.append((r.time, word))

words = pd.DataFrame(rows, columns=['time','word'])

#### remove empty strings #####
words = words[words.word.str.len()>0]

#### stemmer ##### #Number of words = stems so no need
#from nltk.stem.snowball import SnowballStemmer
#stemmer = SnowballStemmer('english')
#stems = [stemmer.stem(t) for t in words['word']]

##### Getting all words #####
totalvocab = words['word'].values.tolist()
vocablist = ' '.join(str(e)for e in totalvocab)
vocablist =[vocablist]

####### TF-IDF with Scikit Learn ######
from sklearn.feature_extraction.text import TfidfVectorizer

# define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer( max_features=200000, stop_words='english', ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(vocablist)
terms = tfidf_vectorizer.get_feature_names


###### Cosine similarity #####
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

##### K-Means Clustering ######
from sklearn.cluster import KMeans
num_clusters = 1
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()


        
    











