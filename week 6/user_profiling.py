#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:11:16 2019
User Profiling
@author: verareyes
"""
##### Import libraries
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import plotly.io as pio
pio.renderers.default = "browser"


##### Load Data
df = pd.read_csv('/home/verareyes/lp/street/database.csv')

##### Getting value counts
user_count = pd.DataFrame(df['user'].value_counts())


##### Distribution of user counts
#user_count['user'].plot(kind='hist')
#print(user_count[:150])

##### List of Target Users
user_count['name'] = user_count.index
users = pd.DataFrame(user_count)
users.reset_index(level=0, inplace=True)
users = users.drop(['index'], axis=1)
users = users.rename(columns={'user':'counts', 'name':'user'})
top_users = users[:105]
low_users = users[105:]
del_users = low_users['user'].tolist()

##### chat types per top user
user_data = df
user_data = user_data[~user_data.user.isin(del_users)]
count = pd.DataFrame(user_data.groupby(['user', 'category']).size(), columns=['number'])
count['group'] = count.index
count.reset_index(level=0, inplace=True)
count.reset_index(level=0, inplace=True)

##### Count of occurence by Category
#y0 = count.loc[count['category'] == 'request']['number']
#y1 = count.loc[count['category'] == 'comment']['number']
#y2 = count.loc[count['category'] == 'caction']['number']
#y3 = count.loc[count['category'] == 'cdesign']['number']
#y4 = count.loc[count['category'] == 'cinfo']['number']
#y5 = count.loc[count['category'] == 'goodp']['number']
#y6 = count.loc[count['category'] == 'badp']['number']
#y7 = count.loc[count['category'] == 'question']['number']
#y8 = count.loc[count['category'] == 'game']['number']
#y9 = count.loc[count['category'] == 'other']['number']
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
#    title = "Chat Type Occurence of Top Users"
#)
#
#fig = go.Figure(data=data,layout=layout)
#fig.show()