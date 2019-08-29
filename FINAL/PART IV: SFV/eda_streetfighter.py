#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:40:48 2019
EDA on street fighter database
@author: verareyes
"""

##### Import libraries
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
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import plotly.io as pio
pio.renderers.default = "browser"

##### Load Data
df = pd.read_csv('/home/verareyes/lp/street/database.csv')

##### Partition and create groups
##### Chat Types / Categories Groups
#chat_types = df
#chat_request = chat_types[chat_types.category == 'request']
#chat_comment = chat_types[chat_types.category == 'comment']
#
#
###### Event Types Groups
#event_types = df
#event_break = event_types[event_types.event == 'break']
#count = event_types.groupby(['event', 'category']).size()
#top_labels = ['bad','ation','design','info','com','game','good<br>move','non-game','quest','req']
##colors = ['rgb(214, 12, 140)','rgb(10, 140, 208)','rgb(10, 140, 208)','rgb(10, 140, 208)','rgb(10, 0, 100)','rgb(100, 0, 10)','rgb(255, 144, 14)','rgb(7,40,89)','rgb(255, 65, 54)','rgb(12, 102, 14)']
#colors = ['rgb(214, 12, 140)','rgb(10, 140, 208)','rgb(10, 140, 208)','rgb(10, 140, 208)','rgb(222, 223, 0)','rgb(100, 0, 10)','rgb(255, 144, 14)','rgb(7,40,89)','rgb(255, 65, 54)','rgb(12, 102, 14)']
# #'rgb(222, 223, 0)'
#x_data = [[0.3,3.6,9.6,14,2,18.7,2.1,31.1,9.6,9],
#          [0.7,4.8,7,10.9,3.1,12.1,2,39.7,11.7,8.1],
#          [0,2.2,7.7,8.8,1.1,26.4,16.5,20.9,9.9,6.6],
#          [0.5,10.4,4,10.9,5.4,16.8,11.4,20.8,8.9,10.9],
#          [5.7,8,3.9,7.3,3.6,8.3,15.5,33.9,6.5,7.3],
#          [1.1,3.5,8.9,12.2,3,15.4,4.2,36.5,9.1,6.1],
#          [1.7,8.6,8.4,11.3,2.5,13.2,7,32.4,9.1,5.7],
#          [1.4,8.4,7.4,11.7,3.8,12.9,5.7,30.2,11,7.5],
#          [0,0,6.3,12.5,6.3,25,0,18.8,25,6.3],
#          [1.4,0.7,5.1,20.3,0.7,24.6,1.4,23.9,15.9,5.8],
#          [2.1,7.2,13.9,25.7,0.4,8.9,3.8,24.9,4.6,8.4],
#          [1,4.2,6.4,9.5,4.2,17.4,5.4,30.9,13.7,7.3],
#          [4.3,4.3,16.3,8.7,0,21.7,6.5,23.9,12,2.2]]
##x_data = [[2,25,67,98,14,131,15,218,67,63],
##          [4,28,41,64,18,71,12,234,69,48],
##          [0,2,7,8,1,24,15,19,9,6],
##          [1,21,8,22,11,34,23,42,18,22],
##          [22,31,15,28,14,32,60,131,25,28],
##          [14,44,111,153,37,193,53,458,114,77],
##          [18,90,88,119,26,139,74,340,96,60],
##          [12,74,65,103,33,113,50,265,97,66],
##          [0,0,1,2,1,4,0,3,4,1],
##          [2,1,7,28,1,34,2,33,22,8],
##          [5,17,33,61,1,21,9,59,11,20],
##          [5,22,33,49,22,90,28,160,71,38],
##          [4,4,15,8,0,20,6,22,11,2]]
#
##x_data = [[0,5,14,20,3,27,3,14,13],
##          [1,8,12,18,5,20,3,19,14],
##          [0,3,10,11,1,33,21,13,8],
##          [1,13,5,14,7,21,14,11,14],
##          [9,12,6,11,5,13,24,10,11],
##          [2,6,14,19,5,24,7,14,10],
##          [3,13,12,17,4,20,10,14,8],
##          [2,12,11,17,5,18,8,16,11],
##          [0,0,8,15,8,31,0,31,8],
##          [2,1,7,27,1,32,2,21,8],
##          [3,10,19,34,1,12,5,6,11],
##          [1,6,9,14,6,25,8,20,11],
##          [6,6,21,11,0,29,9,16,3]]
#y_data = ['before match','break','character data','final round','match lose','round one','round two','match win','player standings','select character','story','training','trials']
#fig = go.Figure()
#for i in range(0, len(x_data[0])):
#    for xd, yd in zip(x_data, y_data):
#        fig.add_trace(go.Bar(
#            x=[xd[i]], y=[yd],
#            orientation='h',
#            marker=dict(
#                color=colors[i],
#                line=dict(color='rgb(248, 248, 249)', width=1)
#            )
#        ))
#
#fig.update_layout(
#    xaxis=dict(
#        showgrid=False,
#        showline=False,
#        showticklabels=False,
#        zeroline=False,
#        domain=[0.15, 1]
#    ),
#    yaxis=dict(
#        showgrid=False,
#        showline=False,
#        showticklabels=False,
#        zeroline=False,
#    ),
#    barmode='stack',
#    paper_bgcolor='rgb(248, 248, 255)',
#    plot_bgcolor='rgb(248, 248, 255)',
#    margin=dict(l=10, r=10, t=100, b=10),
#    showlegend=False,
#)
#
#annotations = []
#
#for yd, xd in zip(y_data, x_data):
#    # labeling the y-axis
#    annotations.append(dict(xref='paper', yref='y',
#                            x=0.14, y=yd,
#                            xanchor='right',
#                            text=str(yd),
#                            font=dict(family='Arial', size=10,
#                                      color='rgb(67, 67, 67)'),
#                            showarrow=False, align='right'))
#    # labeling the first percentage of each bar (x_axis)
#    annotations.append(dict(xref='x', yref='y',
#                            x=xd[0]/2, y=yd,
#                            text=str(xd[0]) + '%',
#                            font=dict(family='Arial', size=10,
#                                      color='rgb(248, 248, 255)'),
#                            showarrow=False))
#    # labeling the first Likert scale (on the top)
#    if yd == y_data[-1]:
#        annotations.append(dict(xref='x', yref='paper',
#                                x=xd[0]/2, y=1.1,
#                                text=top_labels[0],
#                                font=dict(family='Arial', size=10,
#                                          color='rgb(67, 67, 67)'),
#                                showarrow=False))
#    space = xd[0]
#    for i in range(1, len(xd)):
#            # labeling the rest of percentages for each bar (x_axis)
#            annotations.append(dict(xref='x', yref='y',
#                                    x=space + (xd[i]/2), y=yd,
#                                    text=str(xd[i]) + '%',
#                                    font=dict(family='Arial', size=10,
#                                              color='rgb(248, 248, 255)'),
#                                    showarrow=False))
#            # labeling the Likert scale
#            if yd == y_data[-1]:
#                annotations.append(dict(xref='x', yref='paper',
#                                        x=space + (xd[i]/2), y=1.1,
#                                        text=top_labels[i],
#                                        font=dict(family='Arial', size=10,
#                                                  color='rgb(67, 67, 67)'),
#                                        showarrow=False))
#            space += xd[i]
#
#fig.update_layout(annotations=annotations)
#
#fig.show()
##### EDA on Events
##### Percentage of Chat Types per Each Event Category



##### Playing with TextBlob
df['polarity'] = df['message'].map(lambda text: TextBlob(text).sentiment.polarity)  
df['message_len'] = df['message'].astype(str).apply(len)
df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))
user = df.groupby('user').size()
user_df = pd.DataFrame(user)
user_df.to_csv('/home/verareyes/user_count.csv')

##### SENTIMENTS
#print('high positive:\n')
#cl = df.loc[df.polarity == 1, ['message']].sample(5).values
#for c in cl:
#    print(c[0])

#### distribution of sentiment distribution
#df['polarity'].plot(kind='hist')

#### Sentiment by Chat Category
#y0 = df.loc[df['category'] == 'request']['polarity']
#y1 = df.loc[df['category'] == 'comment']['polarity']
#y2 = df.loc[df['category'] == 'caction']['polarity']
#y3 = df.loc[df['category'] == 'cdesign']['polarity']
#y4 = df.loc[df['category'] == 'cinfo']['polarity']
#y5 = df.loc[df['category'] == 'goodp']['polarity']
#y6 = df.loc[df['category'] == 'badp']['polarity']
#y7 = df.loc[df['category'] == 'question']['polarity']
#y8 = df.loc[df['category'] == 'game']['polarity']
#y9 = df.loc[df['category'] == 'other']['polarity']
#
#trace0 = go.Box(
#    y=y0,
#    name = 'request',
#    marker = dict(
#        color = 'rgb(214, 12, 140)',
#    )
#)
#trace1 = go.Box(
#    y=y1,
#    name = 'comment',
#    marker = dict(
#        color = 'rgb(0, 128, 128)',
#    )
#)
#trace2 = go.Box(
#    y=y2,
#    name = 'caction',
#    marker = dict(
#        color = 'rgb(10, 140, 208)',
#    )
#)
#trace3 = go.Box(
#    y=y3,
#    name = 'cdesign',
#    marker = dict(
#        color = 'rgb(12, 102, 14)',
#    )
#)
#trace4 = go.Box(
#    y=y4,
#    name = 'cinfo',
#    marker = dict(
#        color = 'rgb(10, 0, 100)',
#    )
#)
#trace5 = go.Box(
#    y=y5,
#    name = 'goodp',
#    marker = dict(
#        color = 'rgb(100, 0, 10)',
#    )
#
#)
#trace6 = go.Box(
#    y=y6,
#    name = 'badp',
#    marker = dict(
#        color = 'rgb(255, 144, 14)',
#    )
#)
#trace7 = go.Box(
#    y=y7,
#    name = 'question',
#    marker = dict(
#        color = 'rgb(7,40,89)',
#    )
#)
#trace8 = go.Box(
#    y=y8,
#    name = 'game',
#    marker = dict(
#        color = 'rgb(255, 65, 54)',
#    )
#)
#trace9 = go.Box(
#    y=y9,
#    name = 'other',
#    marker = dict(
#        color = 'rgb(222, 223, 0)',
#    )
#)
#data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9]
#layout = go.Layout(
#    title = "Sentiment Polarity Boxplot of Chat Type"
#)
#
#fig = go.Figure(data=data,layout=layout)
#fig.show()

##### Sentiment by Event type
#y0 = df.loc[df['event'] == 'break']['polarity']
#y1 = df.loc[df['event'] == 'training']['polarity']
#y2 = df.loc[df['event'] == 'select character']['polarity']
#y3 = df.loc[df['event'] == 'trials']['polarity']
#y4 = df.loc[df['event'] == 'story']['polarity']
#y5 = df.loc[df['event'] == 'character data']['polarity']
#y6 = df.loc[df['event'] == 'match round one']['polarity']
#y7 = df.loc[df['event'] == 'match round two']['polarity']
#y8 = df.loc[df['event'] == 'final round']['polarity']
#y9 = df.loc[df['event'] == 'match win']['polarity']
#y10 = df.loc[df['event'] == 'match lose']['polarity']
#y11 = df.loc[df['event'] == 'before match']['polarity']
#y12 = df.loc[df['event'] == 'player standings']['polarity']
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
#    title = "Sentiment Polarity Boxplot of Event Types"
#)
#
#fig = go.Figure(data=data,layout=layout)
#fig.show()

##### distribution of message length
#df['message_len'].plot(kind='hist')

##### message length by event
#y0 = df.loc[df['event'] == 'break']['message_len']
#y1 = df.loc[df['event'] == 'training']['message_len']
#y2 = df.loc[df['event'] == 'select character']['message_len']
#y3 = df.loc[df['event'] == 'trials']['message_len']
#y4 = df.loc[df['event'] == 'story']['message_len']
#y5 = df.loc[df['event'] == 'character data']['message_len']
#y6 = df.loc[df['event'] == 'match round one']['message_len']
#y7 = df.loc[df['event'] == 'match round two']['message_len']
#y8 = df.loc[df['event'] == 'final round']['message_len']
#y9 = df.loc[df['event'] == 'match win']['message_len']
#y10 = df.loc[df['event'] == 'match lose']['message_len']
#y11 = df.loc[df['event'] == 'before match']['message_len']
#y12 = df.loc[df['event'] == 'player standings']['polarity']
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
#    title = "Message Length Boxplot of Event Types"
#)
#
#fig = go.Figure(data=data,layout=layout)
#fig.show()

##### Message length by Category
#y0 = df.loc[df['category'] == 'request']['message_len']
#y1 = df.loc[df['category'] == 'comment']['message_len']
#y2 = df.loc[df['category'] == 'caction']['message_len']
#y3 = df.loc[df['category'] == 'cdesign']['message_len']
#y4 = df.loc[df['category'] == 'cinfo']['message_len']
#y5 = df.loc[df['category'] == 'goodp']['message_len']
#y6 = df.loc[df['category'] == 'badp']['message_len']
#y7 = df.loc[df['category'] == 'question']['message_len']
#y8 = df.loc[df['category'] == 'game']['message_len']
#y9 = df.loc[df['category'] == 'other']['message_len']
#
#trace0 = go.Box(
#    y=y0,
#    name = 'request',
#    marker = dict(
#        color = 'rgb(214, 12, 140)',
#    )
#)
#trace1 = go.Box(
#    y=y1,
#    name = 'comment',
#    marker = dict(
#        color = 'rgb(0, 128, 128)',
#    )
#)
#trace2 = go.Box(
#    y=y2,
#    name = 'caction',
#    marker = dict(
#        color = 'rgb(10, 140, 208)',
#    )
#)
#trace3 = go.Box(
#    y=y3,
#    name = 'cdesign',
#    marker = dict(
#        color = 'rgb(12, 102, 14)',
#    )
#)
#trace4 = go.Box(
#    y=y4,
#    name = 'cinfo',
#    marker = dict(
#        color = 'rgb(10, 0, 100)',
#    )
#)
#trace5 = go.Box(
#    y=y5,
#    name = 'goodp',
#    marker = dict(
#        color = 'rgb(100, 0, 10)',
#    )
#
#)
#trace6 = go.Box(
#    y=y6,
#    name = 'badp',
#    marker = dict(
#        color = 'rgb(255, 144, 14)',
#    )
#)
#trace7 = go.Box(
#    y=y7,
#    name = 'question',
#    marker = dict(
#        color = 'rgb(7,40,89)',
#    )
#)
#trace8 = go.Box(
#    y=y8,
#    name = 'game',
#    marker = dict(
#        color = 'rgb(255, 65, 54)',
#    )
#)
#trace9 = go.Box(
#    y=y9,
#    name = 'other',
#    marker = dict(
#        color = 'rgb(222, 223, 0)',
#    )
#)
#data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9]
#layout = go.Layout(
#    title = "Message Length Boxplot of Chat Type"
#)
#
#fig = go.Figure(data=data,layout=layout)
#fig.show()


##### distribution of word count
#df['word_count'].plot(kind='hist')

##### unigrams
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

##### bi-grams
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


##### tri-grams
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

##### POS tagging
#blob = TextBlob(str(df['message']))
#pos_df = pd.DataFrame(blob.tags, columns = ['word', 'pos'])
#pos_df = pos_df.pos.value_counts()[:20]
#pos_df.plot(kind='bar')



    

