#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:28:54 2019
KMeans Clustering (Sentence Level using Word2Vec)
@author: verareyes
"""

####### Import Libraries ########
import nltk
from nltk.cluster import KMeansClusterer
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import sklearn
from sklearn import cluster
from sklearn import metrics
import matplotlib
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import string
import re

####### Read file and Clean Data 
df = df = pd.read_csv('/home/verareyes/lp/street/street_01.txt', sep='[<>]', engine='python', error_bad_lines=False, names=('time','user','message'))
df = df[~df['message'].isnull()]
#######removing messages from bots
data = df
df1 = df.set_index('user')
df1 = df1.drop('Nightbot', axis=0) 
df = df1
###### Clean message 
defined_stop_words = ['pogchamp', 'mike', 'ross', 'daigo', 'ono', 'bakerboy', 'toka', 'vesperspit', 'lul', 'biblethump', 'babyrage', 'kappa', 'ayaya', 'omegalul', 'pepehands', 'pepelaugh', 'lulw', 'weirdeyes', 'nlkripp', 'pog', 'ez', 'smorc', 'lettuce', 'monkaw', 'pogu', 'trihard', 'vespercoffee', 'vespercaptain', 'vespership', 'cawfee', 'cryaya', 'lolw', 'wutface', 'dansgame', 'notlikethis', 'coggers', 'monkas', 'forsenjoy', 'pokmachamp', 'letuce', 'lattuce', 'frankerz', 'toastyc', 'wutfacew', 'poggers', 'jebaited', 'pepega', 'gachibass', 'pogt', 'widehard', 'kreygasm', 'bullyarcade', 'dmjaraxxus', 'biyonzo', 'httpsclipstwitchtvobesegentlenostrilkappaross', 'tsosr', 'krippsniper', 'yoggers', 'lcdoko', 'popoga', 'tylerfree', 'nakkidnf', 'rudehi', 'moonmlady', 'forsenmaldio', 'zubcomfy', 'admiralc', 'reckful', 'pleb', 'danw', 'tylerh', 'blessrng', 'naxx', 'anxiouskitty', 'coolcat', 'novalewut', 'cyrayaya', 'cees', 'inzhmm', 'poggop', 'baboo', 'thijsgasm', 'ppoverheat', 'brofist', 'krippflick', 'champega', 'packsme', 'residentsleeper', 'weirdchamp', 'baaaaby', 'krappa', 'brokeback', 'ayayaweird', 'krippsalt', 'krippready', 'pepejam', 'zulul', 'kkona', 'kripptriggered', 'cmonbruh', 'kapp', 'burself', 'kapp', 'kkop', 'astolfosmile', 'hyperclap', 'jebatied', 'weirdchhamp', 'moshi', 'omegapoggers', 'ratirlcrazy', 'widepeepohappy', 'hypersmorc', 'dududu', 'monkahmm', 'coolstorybob', 'moonsmug', 'xqct', 'amazw', 'eloisee', 'hswp', 'oof', 'pinheapout', 'forsenbee', 'opieop', 'gachigasm', 'neverlucky', 'ikeepitlul', 'yeeeeeeeeeeeeeeeeeeeeeeeee', 'feelsgoodman', 'moonpregario', 'holidaysanta', 'shroudhead', 'doctorwarcry', 'squadw', 'mitchw', 'pjsalt', 'krippa', 'ammorage', 'toastyw', 'savjzw', 'wixmini' 'bogahey', 'sodaayaya', 'lvl', 'elegiggle', 'atpcap', 'anele', 'pepeg', 'scacs', 'squid', 'forsenmald', 'rder', 'gifgoldblum', 'therealbs', 'nipple', 'fugboi',  'bajs', 'moonpregigi', 'mathil', 'poki', 'freehotform', 'kripple', 'monkaeyes', 'asseroth', 'osfrog', 'futanari', 'greekb', 'pefeyga', 'pzayaya', 'plebtest','lulnenyoggers', 'feelsweirdman', 'kriiipyyyyy', 'beardggderp', 'lettuceos', 'rururururu', 'smork', 'vapenation', 'feelsbadman', 'krippstory', 'krippomega', 'mrgrgrgrgrg', 'pokiw', 'rekt', 'vesperarcade', 'vesper', 'smug', 'smugdabeasttv', 'sonicsshiki', 'sonicsol', 'michingmallecho', 'oceanfromblue', 'crazyspanks', 'owltracer', 'jimhey', 'geoffthehero', 'akajoness', 'vohiyo', 'sanford', 'vespermouth']
j = list(string.punctuation) + defined_stop_words
stopwords = j
def preprocess(x):
    x = re.sub('[^a-z\s]','',x.lower()) #remove noise
    x = [w for w in x.split() if w not in set(stopwords)] #remove stopwords
    return ' '.join(x) #join the list

df['message'] = df['message'].apply(preprocess)
df = df.dropna()
df['words'] = df.message.str.strip().str.split('[\W_]+')
####### Pandas to List of Sentences
sentences = df['words'].tolist()
#
#

#sentences = [['ayy'], [''], ['first'], ['ayy', '123'], ['ayy']]

seen = []
sentences = [x for x in sentences if not x in seen and not seen.append(x)]
#sentences = sentences[:100]

####### Training Data ########
#sentences = [['this', 'is', 'the', 'one','good', 'machine', 'learning', 'book'],
#            ['this', 'is',  'another', 'book'],
#            ['one', 'more', 'book'],
#            ['weather', 'rain', 'snow'],
#            ['yesterday', 'weather', 'snow'],
#            ['forecast', 'tomorrow', 'rain', 'snow'],
#            ['this', 'is', 'the', 'new', 'post'],
#            ['this', 'is', 'about', 'more', 'machine', 'learning', 'post'],  
#            ['and', 'this', 'is', 'the', 'one', 'last', 'post', 'book']]
#  
  
 
model = Word2Vec(sentences, min_count=1)
 
  
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
    return np.asarray(sent_vec) / numw
  
  
X=[]
for sentence in sentences:
    X.append(sent_vectorizer(sentence, model))   
 
#print ("========================")
#print (X)
   

#  model[model.vocab] 
#print (model[model.wv.vocab])
#print (model.similarity('post', 'book'))
#print (model.most_similar(positive=['machine'], negative=[], topn=2))

  
NUM_CLUSTERS=15
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
#print (assigned_clusters)
  
  
  
for index, sentence in enumerate(sentences[500:600]):    
    print (str(assigned_clusters[index]) + ":" + str(sentence))
 
     
     
     
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)
  
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
  
#print ("Cluster id labels for inputted data")
#print (labels)
#print ("Centroids data")
#print (centroids)
#  
#print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
#print (kmeans.score(X))
  
silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
  
#print ("Silhouette_score: ")
#print (silhouette_score)
 
#model = TSNE(n_components=3, random_state=0)
#np.set_printoptions(suppress=True)
#Y=model.fit_transform(X)
#plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290,alpha=.5)
# 
#for j in range(len(sentences)):    
#   plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
#   print ("%s %s" % (assigned_clusters[j],  sentences[j]))
#plt.show()



