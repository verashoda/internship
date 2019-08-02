#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:48:26 2019

@author: verareyes
"""
#Creating the frequency distribution of words

import matplotlib.pyplot as plt
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
import csv

file = open("/home/verareyes/twitch_clips/fortnite/fort_01_time.txt", "r")
p = file.read()
fdist = FreqDist()
for sentence in nltk.tokenize.sent_tokenize(p):
    for word in nltk.tokenize.word_tokenize(sentence):
        noise={"[", "]", "<", ">", "the", "to", "``", "a", "you", "?", "it", "!", "me", "and", "TO", "THIS", ":", "SPAM", "is", "HELP", "for", "i", "all", "in", "this", "on", "can", "of", "so", "please", "get", "of", "if", "do", "that", "be", "an", "my", "but", "no", "they", "will", "THE", "are", "at", "I", "'s", "'re", "'ll", "'", ",", ".", "have", "got", "with", "YOU", "your", "(", ")", "we", "’", "was", "A", "ME", "na", "did", "IT", "im", "IS", "IF", "gon", "WE", "'s", "''", "n't", "'m"}
        if word not in noise:
            fdist[word]+=1


import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()
fdist.pprint(800)

dataFrame = pd.DataFrame(list(fdist.items()), columns = ["Time", "Frequency" ])
dataFrame.to_csv('/home/verareyes/twitch_clips/fortnite/fort_01_freq.csv', index=False, header=True)

