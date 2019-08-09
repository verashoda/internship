#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:32:01 2019
Test
@author: verareyes
"""

import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
import re
import numpy as np
import sklearn
import matplotlib
from matplotlib import pyplot as plt


#df = pd.read_csv('/home/verareyes/lp/mine_01.txt', sep='[:,|<>_]', engine='python', error_bad_lines=False, names=('a','b','c','d','e'))

######## Reading txt file to dataframe with multiple delimiters #######
df = pd.read_csv('/home/verareyes/lp/mine_01.txt', sep='[<>]', engine='python', error_bad_lines=False, names=('time','user','message'))
grouped = df.groupby('time')
df1 = pd.DataFrame(df)


##### Stopwords & noise ######
defined_stop_words = ['pogchamp', 'lul', 'biblethump', 'babyrage', 'kappa']
j = list(string.punctuation) + defined_stop_words
stopwords = j
#i = nltk.corpus.stopwords.words('english')
#j = list(string.punctuation) + defined_stop_words
#stopwords = set(i).union(j)

####### Convert to multi-line function ######
def preprocess(x):
    x = re.sub('[^a-z\s]','',x.lower()) #remove noise
    x = [w for w in x.split() if w not in set(stopwords)] #remove stopwords
    return ' '.join(x) #join the list

df1['message'] = df1['message'].apply(preprocess)
#df1 = pd.DataFrame(message)

####### word tokenize ######
df1['words'] = df1.message.str.strip().str.split('[\W_]+')

##### Flatten Data #####
rows = list()
for row in df1[['time', 'words']].iterrows():
    r = row[1]
    for word in r.words:
        rows.append((r.time, word))

words = pd.DataFrame(rows, columns=['time', 'word'])

#### remove empty strings #####
words = words[words.word.str.len()>0]

###### Calculating the TF-IDF ######
words['word'] = words.word.str.lower()
counts = words.groupby('time')\
    .word.value_counts()\
    .to_frame()\
    .rename(columns={'word':'n_w'})

##### visualize the plot #######
#def pretty_plot_top_n (series, top_n=5, index_level=0):
#    r = series\
#    .groupby(level=index_level)\
#    .nlargest(top_n)\
#    .reset_index(level=index_level, drop=True)
#    r.plot.bar()
#    return r.to_frame()
#
#pretty_plot_top_n(counts['n_w'])

###### calculating the TF ######
word_sum = counts.groupby(level=0)\
.sum()\
.rename(columns={'n_w':'n_d'})
word_sum

tf = counts.join(word_sum)
tf['tf'] = tf.n_w/tf.n_d

####### removing stop words #####
c_d = words.time.nunique()

####### Calculating the TD-IF ######

idf = words.groupby('word')\
.time\
.nunique()\
.to_frame()\
.rename(columns={'time':'i_d'})\
.sort_values('i_d')

idf['idf'] = np.log(c_d/idf.i_d.values)
tf_idf = tf.join(idf)
tf_idf['tf_idf'] = tf_idf.tf * tf_idf.idf


###### numerical values of words ######
tokens = tf_idf['tf_idf']
#tokens['time_word'] = tokens.index

##### K-Means Clustering ######
#from sklearn.cluster import KMeans
#num_clusters = 5
#km = KMeans(n_clusters=num_clusters)
#km.fit(tf_idf)
#clusters = km.labels_.tolist()












