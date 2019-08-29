#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:07:32 2019
Personalities Chat Message Per Event
@author: verareyes
"""

import pandas as pd
import numpy as np
import textblob
from textblob import TextBlob
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import plotly.io as pio
pio.renderers.default = "browser"

##### Load data
df = pd.read_csv('/home/verareyes/lp/street/database.csv')

##### Get the users with more than 3 messages
user_count = pd.DataFrame(df['user'].value_counts())
user_count.reset_index(level=0, inplace=True)
user_count = user_count.rename(columns={'user':'counts', 'index':'user'})
low_users = user_count[350:]
del_users = low_users['user'].tolist()
df = df[~df.user.isin(del_users)]

##### Count the type of chat per user and percentage
count = pd.DataFrame(df.groupby(['user', 'category']).size(), columns=['number'])
user = count.groupby(['user']).agg({'number': 'sum'})
percentage = count.div(user, level='user') * 100

##### Get the list of users per personality
percentage.reset_index(level=0, inplace=True)
percentage.reset_index(level=0, inplace=True)
percentage.reset_index(level=0, inplace=True)
groups = percentage



### REQUESTERS
request = groups[(groups.category == 'request') & (groups.number > 25)]
requesters = request['user'].tolist()
requesters_data = df[df.user.isin(requesters)]
req_count = pd.DataFrame(requesters_data.groupby(['event', 'category']).size(), columns=['number'])
##### message length
requesters_data['message_len'] = requesters_data['message'].astype(str).apply(len)
requesters_data['polarity'] = requesters_data['message'].map(lambda text: TextBlob(text).sentiment.polarity) 
####### count the type of chat per event and percentage
#req = req_count.groupby(['event']).agg({'number': 'sum'})
#req_percentage = req_count.div(req, level='event') * 100
###### count the message lenth and percentage

##### unigrams
#def get_top_words(corpus, n=None):
#    vec = CountVectorizer(stop_words = 'english').fit(corpus)
#    bag_of_words = vec.transform(corpus)
#    sum_words = bag_of_words.sum(axis=0)
#    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#    return words_freq[:n]
#
#common_words = get_top_words(requesters_data['message'], 20)
##change input with the dataframe name
##for word, freq in common_words:
##    print(word, freq)
#df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
#df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')

##### bigrams
#def get_top_bigrams(corpus, n=None):
#    vec = CountVectorizer(ngram_range=(2,2), stop_words = 'english').fit(corpus)
#    bag_of_words = vec.transform(corpus)
#    sum_words = bag_of_words.sum(axis=0)
#    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#    return words_freq[:n]
#
#common_words = get_top_bigrams(requesters_data['message'], 20) #change input with the dataframe name
##for word, freq in common_words:
##    print(word, freq)
#df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
#df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')


##### message length by event
#df2 = requesters_data
#y0 = df2.loc[df2['event'] == 'break']['message_len']
#y1 = df2.loc[df2['event'] == 'training']['message_len']
#y2 = df2.loc[df2['event'] == 'select character']['message_len']
#y3 = df2.loc[df2['event'] == 'trials']['message_len']
#y4 = df2.loc[df2['event'] == 'story']['message_len']
#y5 = df2.loc[df2['event'] == 'character data']['message_len']
#y6 = df2.loc[df2['event'] == 'match round one']['message_len']
#y7 = df2.loc[df2['event'] == 'match round two']['message_len']
#y8 = df2.loc[df2['event'] == 'final round']['message_len']
#y9 = df2.loc[df2['event'] == 'match win']['message_len']
#y10 = df2.loc[df2['event'] == 'match lose']['message_len']
#y11 = df2.loc[df2['event'] == 'before match']['message_len']
#y12 = df2.loc[df2['event'] == 'player standings']['polarity']
#
#trace0 = go.Box(
#    y=y0,
#    name = 'break',
#    marker = dict(
#        color = 'rgb(214, 12, 140)',
#    )
#)
#trace1 = go.Box(
#    y=y1,
#    name = 'training',
#    marker = dict(
#        color = 'rgb(0, 128, 128)',
#    )
#)
#trace2 = go.Box(
#    y=y2,
#    name = 'select character',
#    marker = dict(
#        color = 'rgb(10, 140, 208)',
#    )
#)
#trace3 = go.Box(
#    y=y3,
#    name = 'trials',
#    marker = dict(
#        color = 'rgb(12, 102, 14)',
#    )
#)
#trace4 = go.Box(
#    y=y4,
#    name = 'story',
#    marker = dict(
#        color = 'rgb(10, 0, 100)',
#    )
#)
#trace5 = go.Box(
#    y=y5,
#    name = 'character data',
#    marker = dict(
#        color = 'rgb(100, 0, 10)',
#    )
#
#)
#trace6 = go.Box(
#    y=y6,
#    name = 'match round one',
#    marker = dict(
#        color = 'rgb(255, 144, 14)',
#    )
#)
#trace7 = go.Box(
#    y=y7,
#    name = 'match round two',
#    marker = dict(
#        color = 'rgb(7,40,89)',
#    )
#)
#trace8 = go.Box(
#    y=y8,
#    name = 'final round',
#    marker = dict(
#        color = 'rgb(255, 65, 54)',
#    )
#)
#trace9 = go.Box(
#    y=y9,
#    name = 'match win',
#    marker = dict(
#        color = 'rgb(222, 223, 0)',
#    )
#)
#trace10 = go.Box(
#    y=y10,
#    name = 'match lose',
#    marker = dict(
#        color = 'rgb(79, 90, 117)',
#    )
#)
#trace11 = go.Box(
#    y=y11,
#    name = 'before match',
#    marker = dict(
#        color = 'rgb(255, 140, 184)',
#    )
#)
#trace12 = go.Box(
#    y=y12,
#    name = 'player standings',
#    marker = dict(
#        color = 'rgb(127, 96, 0)',
#    )
#)
#data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12]
#layout = go.Layout(
#    title = "Message Length Boxplot of Requesters by Event"
#)
#
#fig = go.Figure(data=data,layout=layout)
#fig.show()

##### trigrams
def get_top_trigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3,3), stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]

common_words = get_top_trigrams(requesters_data['message'], 15) #change input with the dataframe name
#for word, freq in common_words:
#    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')


#
#### COMMENTERS
#comment = groups[(groups.category == 'comment') & (groups.number > 25)]
#commenters = comment['user'].tolist()
#commenters_data = df[df.user.isin(commenters)]
#com_count = pd.DataFrame(commenters_data.groupby(['event', 'category']).size(), columns=['number'])
###### message length
#commenters_data['message_len'] = commenters_data['message'].astype(str).apply(len)
#commenters_data['polarity'] = commenters_data['message'].map(lambda text: TextBlob(text).sentiment.polarity) 
####### count the type of chat per event and percentage
#com = com_count.groupby(['event']).agg({'number': 'sum'})
#com_percentage = com_count.div(com, level='event') * 100
#
#
#
##### CHARACTER FANS
#character1 = groups[(groups.category == 'cinfo') & (groups.number > 25)]
#character2 = groups[(groups.category == 'caction') & (groups.number > 25)]
#character3 = groups[(groups.category == 'cdesign') & (groups.number > 25)]
#characters = character1['user'].tolist() + character2['user'].tolist() + character3['user'].tolist()
#characters = list(set(characters))
#characters_data = df[df.user.isin(characters)]
#character_count = pd.DataFrame(characters_data.groupby(['event', 'category']).size(), columns=['number'])
###### message length
#characters_data['message_len'] = characters_data['message'].astype(str).apply(len) 
#characters_data['polarity'] = characters_data['message'].map(lambda text: TextBlob(text).sentiment.polarity) 
####### count the type of chat per event and percentage
#char = character_count.groupby(['event']).agg({'number': 'sum'})
#char_percentage = character_count.div(char, level='event') * 100
#
##### unigrams
#def get_top_words(corpus, n=None):
#    vec = CountVectorizer(stop_words = 'english').fit(corpus)
#    bag_of_words = vec.transform(corpus)
#    sum_words = bag_of_words.sum(axis=0)
#    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#    return words_freq[:n]
#
#common_words = get_top_words(characters_data['message'], 20)
##change input with the dataframe name
##for word, freq in common_words:
##    print(word, freq)
#df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
#df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')


##### bigrams
#def get_top_bigrams(corpus, n=None):
#    vec = CountVectorizer(ngram_range=(2,2), stop_words = 'english').fit(corpus)
#    bag_of_words = vec.transform(corpus)
#    sum_words = bag_of_words.sum(axis=0)
#    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#    return words_freq[:n]
#
#common_words = get_top_bigrams(characters_data['message'], 20) #change input with the dataframe name
##for word, freq in common_words:
##    print(word, freq)
#df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
#df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')


###### Message Length Boxplot
#df2 = characters_data
#y0 = df2.loc[df2['event'] == 'break']['message_len']
#y1 = df2.loc[df2['event'] == 'training']['message_len']
#y2 = df2.loc[df2['event'] == 'select character']['message_len']
#y3 = df2.loc[df2['event'] == 'trials']['message_len']
#y4 = df2.loc[df2['event'] == 'story']['message_len']
#y5 = df2.loc[df2['event'] == 'character data']['message_len']
#y6 = df2.loc[df2['event'] == 'match round one']['message_len']
#y7 = df2.loc[df2['event'] == 'match round two']['message_len']
#y8 = df2.loc[df2['event'] == 'final round']['message_len']
#y9 = df2.loc[df2['event'] == 'match win']['message_len']
#y10 = df2.loc[df2['event'] == 'match lose']['message_len']
#y11 = df2.loc[df2['event'] == 'before match']['message_len']
#y12 = df2.loc[df2['event'] == 'player standings']['polarity']
#
#trace0 = go.Box(
#    y=y0,
#    name = 'break',
#    marker = dict(
#        color = 'rgb(214, 12, 140)',
#    )
#)
#trace1 = go.Box(
#    y=y1,
#    name = 'training',
#    marker = dict(
#        color = 'rgb(0, 128, 128)',
#    )
#)
#trace2 = go.Box(
#    y=y2,
#    name = 'select character',
#    marker = dict(
#        color = 'rgb(10, 140, 208)',
#    )
#)
#trace3 = go.Box(
#    y=y3,
#    name = 'trials',
#    marker = dict(
#        color = 'rgb(12, 102, 14)',
#    )
#)
#trace4 = go.Box(
#    y=y4,
#    name = 'story',
#    marker = dict(
#        color = 'rgb(10, 0, 100)',
#    )
#)
#trace5 = go.Box(
#    y=y5,
#    name = 'character data',
#    marker = dict(
#        color = 'rgb(100, 0, 10)',
#    )
#
#)
#trace6 = go.Box(
#    y=y6,
#    name = 'match round one',
#    marker = dict(
#        color = 'rgb(255, 144, 14)',
#    )
#)
#trace7 = go.Box(
#    y=y7,
#    name = 'match round two',
#    marker = dict(
#        color = 'rgb(7,40,89)',
#    )
#)
#trace8 = go.Box(
#    y=y8,
#    name = 'final round',
#    marker = dict(
#        color = 'rgb(255, 65, 54)',
#    )
#)
#trace9 = go.Box(
#    y=y9,
#    name = 'match win',
#    marker = dict(
#        color = 'rgb(222, 223, 0)',
#    )
#)
#trace10 = go.Box(
#    y=y10,
#    name = 'match lose',
#    marker = dict(
#        color = 'rgb(79, 90, 117)',
#    )
#)
#trace11 = go.Box(
#    y=y11,
#    name = 'before match',
#    marker = dict(
#        color = 'rgb(255, 140, 184)',
#    )
#)
#trace12 = go.Box(
#    y=y12,
#    name = 'player standings',
#    marker = dict(
#        color = 'rgb(127, 96, 0)',
#    )
#)
#data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12]
#layout = go.Layout(
#    title = "Message Length Boxplot of Character Fans by Event"
#)
#
#fig = go.Figure(data=data,layout=layout)
#fig.show()

###### trigrams
#def get_top_trigrams(corpus, n=None):
#    vec = CountVectorizer(ngram_range=(3,3), stop_words = 'english').fit(corpus)
#    bag_of_words = vec.transform(corpus)
#    sum_words = bag_of_words.sum(axis=0)
#    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#    return words_freq[:n]
#
#common_words = get_top_trigrams(characters_data['message'], 15) #change input with the dataframe name
##for word, freq in common_words:
##    print(word, freq)
#df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
#df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')

#
##### PRAISERS
#praise = groups[(groups.category == 'goodp') & (groups.number > 25)]
#praisers = praise['user'].tolist()
#praisers_data = df[df.user.isin(praisers)]
#praise_count = pd.DataFrame(commenters_data.groupby(['event', 'category']).size(), columns=['number'])
###### message length
#praisers_data['message_len'] = praisers_data['message'].astype(str).apply(len)
#praisers_data['polarity'] = praisers_data['message'].map(lambda text: TextBlob(text).sentiment.polarity) 
####### count the type of chat per event and percentage
#pra = praise_count.groupby(['event']).agg({'number': 'sum'})
#praise_percentage = praise_count.div(pra, level='event') * 100
#
#
##### BOOERS
#boo = groups[(groups.category == 'badp') & (groups.number > 25)]
#booers = boo['user'].tolist()
#booers_data = df[df.user.isin(booers)]
#boo_count = pd.DataFrame(booers_data.groupby(['event', 'category']).size(), columns=['number'])
###### message length
#booers_data['message_len'] = booers_data['message'].astype(str).apply(len)
#booers_data['polarity'] = booers_data['message'].map(lambda text: TextBlob(text).sentiment.polarity) 
###### count the type of chat per event and percentage
#booer = boo_count.groupby(['event']).agg({'number': 'sum'})
#boo_percentage = boo_count.div(booer, level='event') * 100
#
#
##### GAME FANS
#game = groups[(groups.category == 'game') & (groups.number > 25)]
#gamers = game['user'].tolist()
#gamers_data = df[df.user.isin(gamers)]
#gamers_count = pd.DataFrame(gamers_data.groupby(['event', 'category']).size(), columns=['number'])
###### message length
#gamers_data['message_len'] = gamers_data['message'].astype(str).apply(len)
#gamers_data['polarity'] = gamers_data['message'].map(lambda text: TextBlob(text).sentiment.polarity) 
####### count the type of chat per event and percentage
#gam = gamers_count.groupby(['event']).agg({'number': 'sum'})
#game_percentage = gamers_count.div(gam, level='event') * 100

##### unigrams
#def get_top_words(corpus, n=None):
#    vec = CountVectorizer(stop_words = 'english').fit(corpus)
#    bag_of_words = vec.transform(corpus)
#    sum_words = bag_of_words.sum(axis=0)
#    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#    return words_freq[:n]
#
#common_words = get_top_words(gamers_data['message'], 20)
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
#common_words = get_top_bigrams(gamers_data['message'], 20) #change input with the dataframe name
##for word, freq in common_words:
##    print(word, freq)
#df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
#df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')

##### tri-grams
#def get_top_trigrams(corpus, n=None):
#    vec = CountVectorizer(ngram_range=(3,3), stop_words = 'english').fit(corpus)
#    bag_of_words = vec.transform(corpus)
#    sum_words = bag_of_words.sum(axis=0)
#    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#    return words_freq[:n]
#
#common_words = get_top_trigrams(gamers_data['message'], 15) #change input with the dataframe name
##for word, freq in common_words:
##    print(word, freq)
#df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
#df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')


###### Message Length Boxplot
#df2 = gamers_data
#y0 = df2.loc[df2['event'] == 'break']['message_len']
#y1 = df2.loc[df2['event'] == 'training']['message_len']
#y2 = df2.loc[df2['event'] == 'select character']['message_len']
#y3 = df2.loc[df2['event'] == 'trials']['message_len']
#y4 = df2.loc[df2['event'] == 'story']['message_len']
#y5 = df2.loc[df2['event'] == 'character data']['message_len']
#y6 = df2.loc[df2['event'] == 'match round one']['message_len']
#y7 = df2.loc[df2['event'] == 'match round two']['message_len']
#y8 = df2.loc[df2['event'] == 'final round']['message_len']
#y9 = df2.loc[df2['event'] == 'match win']['message_len']
#y10 = df2.loc[df2['event'] == 'match lose']['message_len']
#y11 = df2.loc[df2['event'] == 'before match']['message_len']
#y12 = df2.loc[df2['event'] == 'player standings']['polarity']
#
#trace0 = go.Box(
#    y=y0,
#    name = 'break',
#    marker = dict(
#        color = 'rgb(214, 12, 140)',
#    )
#)
#trace1 = go.Box(
#    y=y1,
#    name = 'training',
#    marker = dict(
#        color = 'rgb(0, 128, 128)',
#    )
#)
#trace2 = go.Box(
#    y=y2,
#    name = 'select character',
#    marker = dict(
#        color = 'rgb(10, 140, 208)',
#    )
#)
#trace3 = go.Box(
#    y=y3,
#    name = 'trials',
#    marker = dict(
#        color = 'rgb(12, 102, 14)',
#    )
#)
#trace4 = go.Box(
#    y=y4,
#    name = 'story',
#    marker = dict(
#        color = 'rgb(10, 0, 100)',
#    )
#)
#trace5 = go.Box(
#    y=y5,
#    name = 'character data',
#    marker = dict(
#        color = 'rgb(100, 0, 10)',
#    )
#
#)
#trace6 = go.Box(
#    y=y6,
#    name = 'match round one',
#    marker = dict(
#        color = 'rgb(255, 144, 14)',
#    )
#)
#trace7 = go.Box(
#    y=y7,
#    name = 'match round two',
#    marker = dict(
#        color = 'rgb(7,40,89)',
#    )
#)
#trace8 = go.Box(
#    y=y8,
#    name = 'final round',
#    marker = dict(
#        color = 'rgb(255, 65, 54)',
#    )
#)
#trace9 = go.Box(
#    y=y9,
#    name = 'match win',
#    marker = dict(
#        color = 'rgb(222, 223, 0)',
#    )
#)
#trace10 = go.Box(
#    y=y10,
#    name = 'match lose',
#    marker = dict(
#        color = 'rgb(79, 90, 117)',
#    )
#)
#trace11 = go.Box(
#    y=y11,
#    name = 'before match',
#    marker = dict(
#        color = 'rgb(255, 140, 184)',
#    )
#)
#trace12 = go.Box(
#    y=y12,
#    name = 'player standings',
#    marker = dict(
#        color = 'rgb(127, 96, 0)',
#    )
#)
#data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12]
#layout = go.Layout(
#    title = "Message Length Boxplot of Game Fans by Event"
#)
#
#fig = go.Figure(data=data,layout=layout)
#fig.show()
