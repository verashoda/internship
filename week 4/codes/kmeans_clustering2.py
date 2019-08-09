#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:12:39 2019
Unsupervised Text Clustering (Version 2)
@author: verareyes
"""

import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
import re
import numpy as np
import sklearn
from sklearn.feature_extraction import text
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib
from matplotlib import pyplot as plt

######## Reading txt file to dataframe with multiple delimiters #######
df = pd.read_csv('/home/verareyes/lp/street/street_04.txt', sep='[<>]', engine='python', error_bad_lines=False, names=('time','user','message'))
#######removing messages from bots
#data = df
#df1 = df.set_index('user')
#df1 = df1.drop('StreamElements', axis=0) 
#df = df1

df1 = pd.DataFrame(df['message'])
defined_stop_words = ['pogchamp', 'mike', 'ross', 'daigo', 'ono', 'bakerboy', 'toka', 'vesperspit', 'lul', 'biblethump', 'babyrage', 'kappa', 'ayaya', 'omegalul', 'pepehands', 'pepelaugh', 'lulw', 'weirdeyes', 'nlkripp', 'pog', 'ez', 'smorc', 'lettuce', 'monkaw', 'pogu', 'trihard', 'vespercoffee', 'vespercaptain', 'vespership', 'cawfee', 'cryaya', 'lolw', 'wutface', 'dansgame', 'notlikethis', 'coggers', 'monkas', 'forsenjoy', 'pokmachamp', 'letuce', 'lattuce', 'frankerz', 'toastyc', 'wutfacew', 'poggers', 'jebaited', 'pepega', 'gachibass', 'pogt', 'widehard', 'kreygasm', 'bullyarcade', 'dmjaraxxus', 'biyonzo', 'httpsclipstwitchtvobesegentlenostrilkappaross', 'tsosr', 'krippsniper', 'yoggers', 'lcdoko', 'popoga', 'tylerfree', 'nakkidnf', 'rudehi', 'moonmlady', 'forsenmaldio', 'zubcomfy', 'admiralc', 'reckful', 'pleb', 'danw', 'tylerh', 'blessrng', 'naxx', 'anxiouskitty', 'coolcat', 'novalewut', 'cyrayaya', 'cees', 'inzhmm', 'poggop', 'baboo', 'thijsgasm', 'ppoverheat', 'brofist', 'krippflick', 'champega', 'packsme', 'residentsleeper', 'weirdchamp', 'baaaaby', 'krappa', 'brokeback', 'ayayaweird', 'krippsalt', 'krippready', 'pepejam', 'zulul', 'kkona', 'kripptriggered', 'cmonbruh', 'kapp', 'burself', 'kapp', 'kkop', 'astolfosmile', 'hyperclap', 'jebatied', 'weirdchhamp', 'moshi', 'omegapoggers', 'ratirlcrazy', 'widepeepohappy', 'hypersmorc', 'dududu', 'monkahmm', 'coolstorybob', 'moonsmug', 'xqct', 'amazw', 'eloisee', 'hswp', 'oof', 'pinheapout', 'forsenbee', 'opieop', 'gachigasm', 'neverlucky', 'ikeepitlul', 'yeeeeeeeeeeeeeeeeeeeeeeeee', 'feelsgoodman', 'moonpregario', 'holidaysanta', 'shroudhead', 'doctorwarcry', 'squadw', 'mitchw', 'pjsalt', 'krippa', 'ammorage', 'toastyw', 'savjzw', 'wixmini' 'bogahey', 'sodaayaya', 'lvl', 'elegiggle', 'atpcap', 'anele', 'pepeg', 'scacs', 'squid', 'forsenmald', 'rder', 'gifgoldblum', 'therealbs', 'nipple', 'fugboi',  'bajs', 'moonpregigi', 'mathil', 'poki', 'freehotform', 'kripple', 'monkaeyes', 'asseroth', 'osfrog', 'futanari', 'greekb', 'pefeyga', 'pzayaya', 'plebtest','lulnenyoggers', 'feelsweirdman', 'kriiipyyyyy', 'beardggderp', 'lettuceos', 'rururururu', 'smork', 'vapenation', 'feelsbadman', 'krippstory', 'krippomega', 'mrgrgrgrgrg', 'pokiw', 'rekt', 'vesperarcade', 'vesper', 'smug', 'smugdabeasttv', 'sonicsshiki', 'sonicsol', 'michingmallecho', 'oceanfromblue', 'crazyspanks', 'owltracer', 'jimhey', 'geoffthehero', 'akajoness', 'vohiyo', 'sanford']
j = list(string.punctuation) + defined_stop_words
stopwords = j
def preprocess(x):
    x = re.sub('[^a-z\s]','',x.lower()) #remove noise
    x = [w for w in x.split() if w not in set(stopwords)] #remove stopwords
    return ' '.join(x) #join the list

df1['message'] = df1['message'].apply(preprocess)

data = pd.DataFrame(df1['message'])



data = data.drop_duplicates('message')

######### TF-IDF #########
punc = ['.',',','"','?','!',':',';','(',')','[',']','{','}','%','@']
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
desc = data['message'].values
vectorizer = TfidfVectorizer(stop_words = stop_words)
X = vectorizer.fit_transform(desc)
words = vectorizer.get_feature_names()

#from nltk.stem.snowball import SnowballStemmer
#from nltk.tokenize import RegexpTokenizer
#
#stemmer = SnowballStemmer('english')
#tokenizer = RegexpTokenizer('r[a-zA-Z\']+')
#
#def tokenize(text):
#    return[stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]
    
#vectorizer2 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize)
#X2 = vectorizer2.fit_transform(desc)
#words = vectorizer2.get_feature_names()

#vectorizer3 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize, max_features=1000)
#X3 = vectorizer3.fit_transform(desc)
#words = vectorizer3.get_feature_names()

####### KMeans Elbow Method ##########
#wcss = []
#for i in range(1,21):
#    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#    kmeans.fit(X)
#    wcss.append(kmeans.inertia_)
#plt.plot(range(1,21), wcss)
#plt.title('The Elbow Method')
#plt.show()


#### KMeans Clustering #####

### 5  clusters #####
kmeans = KMeans(n_clusters = 11, n_init=10, n_jobs=1)
kmeans.fit(X)
common_words = kmeans.cluster_centers_.argsort()[:,-1:-10:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))



#
###### Stopwords & noise ######
#defined_stop_words = ['pogchamp', 'lul', 'biblethump', 'babyrage', 'kappa', 'kreygasm']
#j = list(string.punctuation) + defined_stop_words
#stopwords = j
#
######## Convert to multi-line function ######
#def preprocess(x):
#    x = re.sub('[^a-z\s]','',x.lower()) #remove noise
#    x = [w for w in x.split() if w not in set(stopwords)] #remove stopwords
#    return ' '.join(x) #join the list
#
#df1['message'] = df1['message'].apply(preprocess)
##df1 = pd.DataFrame(message)
#
######## word tokenize ######
#df1['words'] = df1.message.str.strip().str.split('[\W_]+')
#
###### Flatten Data #####
#rows = list()
#for row in df1[['time', 'words']].iterrows():
#    r = row[1]
#    for word in r.words:
#        rows.append((r.time, word))
#
#words = pd.DataFrame(rows, columns=['time', 'word'])
#
##### remove empty strings #####
#words = words[words.word.str.len()>0]
#
#
######## TF-IDF ###################
#from sklearn.feature_extraction.text import TfidfVectorizer
#all_vocab = words['word'].values.tolist()
#text = ' '.join(str(e) for e in all_vocab)
#text = [text]
#vectorizer = TfidfVectorizer(max_features=200000)
#X = vectorizer.fit_transform(text)
#word_features = vectorizer.get_feature_names()
#
######## Clustering #########
#wcss = []
#for i in range(1,11):
#    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#    kmeans.fit(X)
#    wcss.append(kmeans.inertia_)
#plt.plot(range(1,11), wcss)
#plt.title('The Elbow Method')
#plt.show()




