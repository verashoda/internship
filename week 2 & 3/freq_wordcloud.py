#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:35:56 2019
Word frequency and world cloud
@author: verareyes
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import seaborn as sns
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib
from matplotlib import pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
#import umap

df = pd.read_csv('/home/verareyes/twitch_clips/fortnite/fort_01_excitedchat.csv')
df1 = pd.DataFrame(df, columns=['chat'])
lists = df1.values.tolist()
text = ''.join(str(e) for e in lists)

from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)

from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
filtered_word=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_word.append(w)
noise={"[", "]", "?", "'", "'", ":", "(", ".", ")", "!", "``", "''", "THIS", "TO", "THE", "allowed"}
final_words=[]
for w in filtered_word:
    if w not in noise:
        final_words.append(w)

words = ''.join(str(e) for e in final_words)


#Creating the frequency distribution of words
#from nltk.probability import FreqDist
#fdist = FreqDist(final_words)
#
#import matplotlib.pyplot as plt
#fdist.plot(40,cumulative=False)
#plt.show()

stop_words = set(STOPWORDS)
stop_words.update(["PogChampPogChampPogChampPogChamp", "BibleThumpBibleThumpBibleThumpBibleThump", "PogChampPogChampPogChamp"])

word_cloud = WordCloud(stopwords=stop_words, background_color="white").generate(words)
plt.figure()
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

word_cloud.to_file('/home/verareyes/twitch_clips/fortnite/fort_01_wordcloud2.png')
