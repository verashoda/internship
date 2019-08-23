#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:50:41 2019
Unigrams Bigrams Trigrams
@author: verareyes
"""

##### Importing libraries
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
##### Loading Data
df = pd.read_csv('/home/verareyes/lp/street/database.csv')

##### Chat Types / Categories Groups
chat_types = df
chat_request = chat_types[chat_types.category == 'request']
chat_comment = chat_types[chat_types.category == 'comment']


##### unigrams
#def get_top_words(corpus, n=None):
#    vec = CountVectorizer(stop_words = 'english').fit(corpus)
#    bag_of_words = vec.transform(corpus)
#    sum_words = bag_of_words.sum(axis=0)
#    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#    return words_freq[:n]
#
#common_words = get_top_words(chat_request['message'], 20)
##change input with the dataframe name
##for word, freq in common_words:
##    print(word, freq)
#df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
#df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')

##### bi-grams
#def get_top_bigrams(corpus, n=None):
#    vec = CountVectorizer(ngram_range=(2,2), stop_words = 'english').fit(corpus)
#    bag_of_words = vec.transform(corpus)
#    sum_words = bag_of_words.sum(axis=0)
#    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#    return words_freq[:n]
#
#common_words = get_top_bigrams(chat_request['message'], 20) #change input with the dataframe name
##for word, freq in common_words:
##    print(word, freq)
#df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
#df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')


##### tri-grams
def get_top_trigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3,3), stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]

common_words = get_top_trigrams(chat_request['message'], 20) #change input with the dataframe name
#for word, freq in common_words:
#    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')

