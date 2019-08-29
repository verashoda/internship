#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:01:03 2019
Data Processing and Cleaning
@author: verareyes
"""

##### Import Libraries
import pandas as pd
import pandas as pd
import string 
import re
import textblob
from textblob import TextBlob

##### Load Data
df = df = pd.read_csv('/home/verareyes/lp/txt/street_01.txt', sep='[<>]', engine='python', error_bad_lines=False, names=('time','user','message'))  #Input path file
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

######## TextBlob #########
df['polarity'] = df['message'].map(lambda text: TextBlob(text).sentiment.polarity)  
df['message_len'] = df['message'].astype(str).apply(len)
df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))

##### Export to csv (initial database)
df.to_csv('/home/verareyes/database.csv')