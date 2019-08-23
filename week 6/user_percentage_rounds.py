#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:11:36 2019
Number of users per round 
@author: verareyes
"""
import pandas as pd
import textblob
from textblob import TextBlob
import matplotlib
from matplotlib import pyplot as plt
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

##### USER COUNT
#r1_w = pd.read_csv('/home/verareyes/lp/street/r2_win_groups.csv')
#r1_wcount = pd.DataFrame(r1_w.groupby([ 'parts','user']).size(), columns=['number'])
#r1_wcount['index1']= r1_wcount.index
#r1_wcount_2 = pd.DataFrame(r1_wcount.groupby(['index1']).size(), columns=['size'])

##### CHAT COUNT
#r1_w = pd.read_csv('/home/verareyes/lp/street/r2_win_groups.csv')
#r1_wcount = pd.DataFrame(r1_w.groupby([ 'parts','user']).size(), columns=['number'])
#r1_wcount.reset_index(level=0, inplace=True)
#r1_wcount.reset_index(level=0, inplace=True)
#r1_w_user_group_by = r1_wcount.groupby(['parts','user']).agg({'number':'sum'})
#r1_w_parts = r1_wcount.groupby(['parts']).agg({'number': 'sum'})

##### MESSAGE LENGTH
#r1_w = pd.read_csv('/home/verareyes/lp/street/r1_lose_groups.csv')
######### Playing with TextBlob #########
#r1_w['polarity'] = r1_w['message'].map(lambda text: TextBlob(text).sentiment.polarity)  
#r1_w['message_len'] = r1_w['message'].astype(str).apply(len)
#r1_w_user_group_by = r1_w.groupby(['parts']).agg({'message_len':'sum'})
###### message length by event box plot
#y0 = r1_w.loc[r1_w['parts'] == 'a']['message_len']
#y1 = r1_w.loc[r1_w['parts'] == 'b']['message_len']
#y2 = r1_w.loc[r1_w['parts'] == 'c']['message_len']
#trace0 = go.Box(
#    y=y0,
#    name = 'start',
#    marker = dict(
#        color = 'rgb(214, 12, 140)',
#    )
#)
#trace1 = go.Box(
#    y=y1,
#    name = 'middle',
#    marker = dict(
#        color = 'rgb(0, 128, 128)',
#    )
#)
#trace2 = go.Box(
#    y=y2,
#    name = 'end',
#    marker = dict(
#        color = 'rgb(10, 140, 208)',
#        )
#)
#data = [trace0, trace1, trace2]
#layout = go.Layout(
#    title = "Message Length Boxplot of ROUND 1 WIN"
#)
#
#fig = go.Figure(data=data,layout=layout)
#fig.show()

##### SENTIMENT
#r1_w = pd.read_csv('/home/verareyes/lp/street/r2_win_groups.csv')
######### Playing with TextBlob ######### 
#r1_w_user_group_by = r1_w.groupby(['parts']).agg({'polarity':'mean'})



#### Sentiment by Rounds Box Plot
r1_w = pd.read_csv('/home/verareyes/lp/street/r1_win_groups.csv')
r1_w['polarity'] = r1_w['message'].map(lambda text: TextBlob(text).sentiment.polarity) 

r1_l = pd.read_csv('/home/verareyes/lp/street/r1_lose_groups.csv')
r1_l['polarity'] = r1_l['message'].map(lambda text: TextBlob(text).sentiment.polarity) 

r2_w = pd.read_csv('/home/verareyes/lp/street/r2_win_groups.csv')
r2_w['polarity'] = r2_w['message'].map(lambda text: TextBlob(text).sentiment.polarity) 

r2_l = pd.read_csv('/home/verareyes/lp/street/r2_lose_groups.csv')
r2_l['polarity'] = r2_l['message'].map(lambda text: TextBlob(text).sentiment.polarity) 

y0 = r1_w.loc[r1_w['parts'] == 'a']['polarity']
y1 = r1_w.loc[r1_w['parts'] == 'b']['polarity']
y2 = r1_w.loc[r1_w['parts'] == 'c']['polarity']
y3 = r1_l.loc[r1_l['parts'] == 'a']['polarity']
y4 = r1_l.loc[r1_l['parts'] == 'b']['polarity']
y5 = r1_l.loc[r1_l['parts'] == 'c']['polarity']
y6 = r2_w.loc[r2_w['parts'] == 'a']['polarity']
y7 = r2_w.loc[r2_w['parts'] == 'b']['polarity']
y8 = r2_w.loc[r2_w['parts'] == 'c']['polarity']
y9 = r2_l.loc[r2_l['parts'] == 'a']['polarity']
y10 = r2_l.loc[r2_l['parts'] == 'b']['polarity']
y11 = r2_l.loc[r2_l['parts'] == 'c']['polarity']
trace0 = go.Box(
    y=y0,
    name = 'r1_w_start',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
trace1 = go.Box(
    y=y1,
    name = 'r1_w_middle',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
trace2 = go.Box(
    y=y2,
    name = 'r1_w_end',
    marker = dict(
        color = 'rgb(10, 140, 208)',
    )
)
trace3 = go.Box(
    y=y3,
    name = 'r1_l_start',
    marker = dict(
        color = 'rgb(12, 102, 14)',
    )
)
trace4 = go.Box(
    y=y4,
    name = 'r1_l_middle',
    marker = dict(
        color = 'rgb(10, 0, 100)',
    )
)
trace5 = go.Box(
    y=y5,
    name = 'r1_l_end',
    marker = dict(
        color = 'rgb(100, 0, 10)',
    )

)
trace6 = go.Box(
    y=y6,
    name = 'r2_w_start',
    marker = dict(
        color = 'rgb(255, 144, 14)',
    )
)
trace7 = go.Box(
    y=y7,
    name = 'r2_w_middle',
    marker = dict(
        color = 'rgb(7,40,89)',
    )
)
trace8 = go.Box(
    y=y8,
    name = 'r2_w_end',
    marker = dict(
        color = 'rgb(255, 65, 54)',
    )
)
trace9 = go.Box(
    y=y9,
    name = 'r2_l_start',
    marker = dict(
        color = 'rgb(222, 223, 0)',
    )
)
trace10 = go.Box(
    y=y10,
    name = 'r2_l_middle',
    marker = dict(
        color = 'rgb(79, 90, 117)',
    )
)
trace11 = go.Box(
    y=y11,
    name = 'r2_l_end',
    marker = dict(
        color = 'rgb(255, 140, 184)',
    )
)
data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11]
layout = go.Layout(
    title = "Sentiment Polarity Boxplot of Rounds"
)

fig = go.Figure(data=data,layout=layout)
fig.show()