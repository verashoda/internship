#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:48:16 2019
EDA 
@author: verareyes
"""

import pandas as pd
import string 
import re
import textblob
from textblob import TextBlob
import matplotlib
from matplotlib import pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import scattertext as st
import spacy
import pprint
from pprint import pprint
import numpy as np
from collections import Counter

#####  Loading the Data  #####
df = df = pd.read_csv('/home/verareyes/lp/txt/street_01.txt', sep='[<>]', engine='python', error_bad_lines=False, names=('time','user','message'))
df = df[~df['message'].isnull()]
#######removing messages from bots
#data = df
#df1 = df.set_index('user')
#df1 = df1.drop('StreamElements', axis=0) 
#df = df1

###### Cleaning data ##########
defined_stop_words = ['pogchamp', 'lul', 'biblethump', 'babyrage', 'kappa', 'ayaya', 'omegalul', 'pepehands', 'pepelaugh', 'lulw', 'weirdeyes', 'nlkripp', 'pog', 'ez', 'smorc', 'lettuce', 'monkaw', 'pogu', 'trihard', 'cryaya', 'lolw', 'wutface', 'dansgame', 'notlikethis', 'coggers', 'monkas', 'forsenjoy', 'pokmachamp', 'letuce', 'lattuce', 'frankerz', 'toastyc', 'wutfacew', 'poggers', 'jebaited', 'pepega', 'gachibass', 'pogt', 'widehard', 'kreygasm', 'httpsclipstwitchtvobesegentlenostrilkappaross', 'tsosr', 'krippsniper', 'yoggers', 'popoga', 'tylerfree', 'nakkidnf', 'rudehi', 'moonmlady', 'forsenmaldio', 'zubcomfy', 'admiralc', 'reckful', 'pleb', 'danw', 'tylerh', 'blessrng', 'naxx', 'anxiouskitty', 'coolcat', 'novalewut', 'cyrayaya', 'cees', 'inzhmm', 'poggop', 'baboo', 'thijsgasm', 'ppoverheat', 'brofist', 'krippflick', 'champega', 'packsme', 'residentsleeper', 'weirdchamp', 'baaaaby', 'krappa', 'brokeback', 'ayayaweird', 'krippsalt', 'krippready', 'pepejam', 'zulul', 'kkona', 'kripptriggered', 'cmonbruh', 'kapp', 'burself', 'kapp', 'kkop', 'astolfosmile', 'hyperclap', 'jebatied', 'weirdchhamp', 'moshi', 'omegapoggers', 'ratirlcrazy', 'widepeepohappy', 'hypersmorc', 'dududu', 'monkahmm', 'coolstorybob', 'moonsmug', 'xqct', 'amazw', 'eloisee', 'hswp', 'oof', 'pinheapout', 'forsenbee', 'opieop', 'gachigasm', 'neverlucky', 'ikeepitlul', 'yeeeeeeeeeeeeeeeeeeeeeeeee', 'feelsgoodman', 'moonpregario', 'holidaysanta', 'shroudhead', 'doctorwarcry', 'squadw', 'mitchw', 'pjsalt', 'krippa', 'ammorage', 'toastyw', 'savjzw', 'wixmini' 'bogahey', 'sodaayaya', 'lvl', 'elegiggle', 'atpcap', 'anele', 'pepeg', 'scacs', 'squid', 'forsenmald', 'rder', 'gifgoldblum', 'therealbs', 'nipple', 'fugboi',  'bajs', 'moonpregigi', 'mathil', 'poki', 'freehotform', 'kripple', 'monkaeyes', 'asseroth', 'osfrog', 'futanari', 'greekb', 'pefeyga', 'pzayaya', 'plebtest','lulnenyoggers', 'feelsweirdman', 'kriiipyyyyy', 'beardggderp', 'lettuceos', 'rururururu', 'smork', 'vapenation', 'feelsbadman', 'krippstory', 'krippomega', 'mrgrgrgrgrg', 'pokiw', 'rekt']
j = list(string.punctuation) + defined_stop_words
stopwords = j
def preprocess(x):
    x = re.sub('[^a-z\s]','',x.lower()) #remove noise
    x = [w for w in x.split() if w not in set(stopwords)] #remove stopwords
    return ' '.join(x) #join the list

df['message'] = df['message'].apply(preprocess)

######## Playing with TextBlob #########
df['polarity'] = df['message'].map(lambda text: TextBlob(text).sentiment.polarity)  
df['message_len'] = df['message'].astype(str).apply(len)
df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))

#### Testing the sentiment #####
#print('high positive:\n')
#cl = df.loc[df.polarity == -1, ['message']].sample(5).values
#for c in cl:
#    print(c[0])
    
#### distribution of sentiment distribution #####
#df['polarity'].plot(kind='hist')

#### distribution of message length ####
#df['message_len'].plot(kind='hist')

#### distribution of word count ####
#df['word_count'].plot(kind='hist')

####### unigrams, bi-grams, tri-grams ######
#def get_top_words(corpus, n=None):
#    vec = CountVectorizer(stop_words = 'english').fit(corpus)
#    bag_of_words = vec.transform(corpus)
#    sum_words = bag_of_words.sum(axis=0)
#    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#    return words_freq[:n]
#
#common_words = get_top_words(df['message'], 20)
#for word, freq in common_words:
#    print(word, freq)
#df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
#df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')

####### bi-grams #######
#def get_top_bigrams(corpus, n=None):
#    vec = CountVectorizer(ngram_range=(2,2), stop_words = 'english').fit(corpus)
#    bag_of_words = vec.transform(corpus)
#    sum_words = bag_of_words.sum(axis=0)
#    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#    return words_freq[:n]
#
#common_words = get_top_bigrams(df['message'], 20)
#for word, freq in common_words:
#    print(word, freq)
#df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
#df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')

######## tri-grams ########
#def get_top_trigrams(corpus, n=None):
#    vec = CountVectorizer(ngram_range=(3,3), stop_words = 'english').fit(corpus)
#    bag_of_words = vec.transform(corpus)
#    sum_words = bag_of_words.sum(axis=0)
#    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#    return words_freq[:n]
#
#common_words = get_top_trigrams(df['message'], 20)
#for word, freq in common_words:
#    print(word, freq)
#df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
#df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')


############ POS ################
#blob = TextBlob(str(df['message']))
#pos_df = pd.DataFrame(blob.tags, columns = ['word', 'pos'])
#pos_df = pos_df.pos.value_counts()[:20]
#pos_df.plot(kind='bar')


######## Latent Semantic Analysis (LSA) for topic modelling #####
reindexed_data = df['message']
tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
reindexed_data = reindexed_data.values
document_term_matrix = tfidf_vectorizer.fit_transform(reindexed_data)
n_topics = 7
lsa_model = TruncatedSVD(n_components=n_topics)
lsa_topic_matrix = lsa_model.fit_transform(document_term_matrix)

def get_keys(topic_matrix):
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys

def keys_to_counts(keys):
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)

lsa_keys = get_keys(lsa_topic_matrix)
lsa_categories, lsa_counts = keys_to_counts(lsa_keys)

def get_top_words(n, keys, document_term_matrix, tfidf_vectorizer):
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
        top_word_indices.append(top_n_word_indices)
    top_words =[]
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1, document_term_matrix.shape[1]))
            temp_word_vector[:, index]=1
            the_word = tfidf_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))
    return top_words

top_words_lsa = get_top_words(10, lsa_keys, document_term_matrix, tfidf_vectorizer)
    
for i in range(len(top_words_lsa)):
    print("Topic {}: ".format(i+1), top_words_lsa[i])



