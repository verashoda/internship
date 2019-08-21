#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:33:27 2019
Round 1 and 2 Further Analysis
@author: verareyes
"""
##### Import libraries
import pandas as pd

##### Load data
df = pd.read_csv('/home/verareyes/lp/street/database.csv')
df1 = df[(df.event == 'match round one') | (df.event == 'match round two') | (df.event == 'match lose') | (df.event == 'match win')]
#df1.to_csv('/home/verareyes/lp/street/rounds.csv')

##### Breakdown into 4 parts 

##### ROUND ONE
##### WIN
#r1_win = df1[(df1.event == 'match round one') | (df1.event == 'match win')]
#r1_win.reset_index(level=0, inplace=True)
#r1_win['diff']= r1_win['index'].diff()
#r1_win = r1_win.fillna(1)
#r1_win.to_csv('/home/verareyes/lp/street/r1_win2.csv')
r1_win = pd.read_csv('/home/verareyes/lp/street/r1_win_data.csv')
r1_win['diff']= r1_win['index'].diff()
r1_win = r1_win.fillna(1)
#r1_win.to_csv('/home/verareyes/lp/street/r1_win.csv')

###### Breakdown event into 3 parts
r1_w = pd.read_csv('/home/verareyes/lp/street/r1_win.csv')
groups = r1_w.groupby(['group']).size()/3
#r1_wcount = pd.DataFrame(r1_w.groupby([ 'parts','category']).size(), columns=['number'])
#r1_wcount.reset_index(level=0, inplace=True)
#r1_wcount.reset_index(level=0, inplace=True)
#r1_w_user_group_by = r1_wcount.groupby(['parts','category']).agg({'number':'sum'})
#r1_w_parts = r1_wcount.groupby(['parts']).agg({'number': 'sum'})
#r1_w_percentage = r1_w_user_group_by.div(r1_w_parts, level='parts') * 100

##### LOSE
r1_lose = df1[(df1.event == 'match round one') | (df1.event == 'match lose')]
r1_lose.reset_index(level=0, inplace=True)
r1_lose['diff']= r1_lose['index'].diff()
r1_lose = r1_lose.fillna(1)
r1_lose = r1_lose[(r1_lose['diff'] == 1)]
#r1_lose = r1_lose[(r1_lose.event == 'match round one')]
#r1_lose['diff2']= r1_lose['index'].diff()
#r1_lose = r1_lose.fillna(1)
#r1_lose.to_csv('/home/verareyes/lp/street/r1_lose.csv')



##### ROUND TWO
##### WIN
#r2_win = df1[(df1.event == 'match round two') | (df1.event == 'match win')]
#r2_win.reset_index(level=0, inplace=True)
#r2_win['diff']= r2_win['index'].diff()
#r2_win = r2_win.fillna(1)
#r2_win = r2_win[(r2_win['diff'] == 1)]
#r2_win = r2_win.drop(['index','diff'],axis =1)
#r2_win = r2_win[(r2_win.event == 'match round two')]
#r1_win['diff2']= r1_win['index'].diff()
#r1_win = r1_win.fillna(1)
#r1_win.to_csv('/home/verareyes/lp/street/r1_win.csv')


##### LOSE
#r2_lose = df1[(df1.event == 'match round two') | (df1.event == 'match lose')]
#r2_lose.reset_index(level=0, inplace=True)
#r2_lose['diff']= r2_lose['index'].diff()
#r2_lose = r2_lose.fillna(1)
#r2_lose = r2_lose[(r2_lose['diff'] == 1)]
#r2_lose = r2_lose.drop(['index','diff'],axis =1)
#r2_lose = r2_lose[(r2_lose.event == 'match round two')]
#r1_win['diff2']= r1_win['index'].diff()
#r1_win = r1_win.fillna(1)
#r1_win.to_csv('/home/verareyes/lp/street/r1_win.csv')
