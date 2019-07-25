#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:09:41 2019

@author: verareyes
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.set_option("display.max_colwidth", 200)
import nltk
import umap


from nltk.tokenize import word_tokenize
text= """ 
 sentence here
"""
tokenized_word=word_tokenize(text)

#Cleaning data by removing stop words
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
filtered_word=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_word.append(w)
noise={"'s", "I", "n't", "'re", "'ll", "'", ",", ".", "'m"}
final_words=[]
for w in filtered_word:
    if w not in noise:
        final_words.append(w)

#Creating the frequency distribution of words
#from nltk.probability import FreqDist
#fdist = FreqDist(final_words)
#
#import matplotlib.pyplot as plt
#fdist.plot(30,cumulative=False)
#plt.show()

#Document-Term Matrix
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', 
max_features= 10000, # keep top 1000 terms 
max_df = 0.5, 
smooth_idf=True)

X = vectorizer.fit_transform(final_words)

#Topic modellling
from sklearn.decomposition import TruncatedSVD

# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=100, random_state=122)

svd_model.fit(X)

len(svd_model.components_)

terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    print("Topic "+str(i)+": ")
    for t in sorted_terms:
        print(t[0])
        print(" ")








