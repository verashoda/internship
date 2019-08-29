#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:24:46 2019
Percentage Change per round
@author: verareyes
"""
import pandas as pd

##### ROUND ONE
##### WIN
r1_w = pd.read_csv('/home/verareyes/lp/street/r1_win_groups.csv')
r1_wcount = pd.DataFrame(r1_w.groupby([ 'parts','category']).size(), columns=['number'])
r1_wcount.reset_index(level=0, inplace=True)
r1_wcount.reset_index(level=0, inplace=True)
r1_w_user_group_by = r1_wcount.groupby(['parts','category']).agg({'number':'sum'})
r1_w_parts = r1_wcount.groupby(['parts']).agg({'number': 'sum'})
r1_w_percentage = r1_w_user_group_by.div(r1_w_parts, level='parts') * 100
r1_w_percentage.reset_index(level=0, inplace=True)
r1_w_percentage.to_csv('/home/verareyes/lp/street/analysis/r1_w_percentage.csv')



##### LOSE
r1_l = pd.read_csv('/home/verareyes/lp/street/r1_lose_groups.csv')
r1_lcount = pd.DataFrame(r1_l.groupby([ 'parts','category']).size(), columns=['number'])
r1_lcount.reset_index(level=0, inplace=True)
r1_lcount.reset_index(level=0, inplace=True)
r1_l_user_group_by = r1_lcount.groupby(['parts','category']).agg({'number':'sum'})
r1_l_parts = r1_lcount.groupby(['parts']).agg({'number': 'sum'})
r1_l_percentage = r1_l_user_group_by.div(r1_l_parts, level='parts') * 100
r1_l_percentage.reset_index(level=0, inplace=True)
r1_l_percentage.to_csv('/home/verareyes/lp/street/analysis/r1_l_percentage.csv')


##### ROUND TWO
##### WIN
r2_w = pd.read_csv('/home/verareyes/lp/street/r2_win_groups.csv')
r2_wcount = pd.DataFrame(r2_w.groupby([ 'parts','category']).size(), columns=['number'])
r2_wcount.reset_index(level=0, inplace=True)
r2_wcount.reset_index(level=0, inplace=True)
r2_w_user_group_by = r2_wcount.groupby(['parts','category']).agg({'number':'sum'})
r2_w_parts = r2_wcount.groupby(['parts']).agg({'number': 'sum'})
r2_w_percentage = r2_w_user_group_by.div(r2_w_parts, level='parts') * 100
r2_w_percentage.reset_index(level=0, inplace=True)
r2_w_percentage.to_csv('/home/verareyes/lp/street/analysis/r2_w_percentage.csv')

##### LOSE
r2_l = pd.read_csv('/home/verareyes/lp/street/r2_lose_groups.csv')
r2_lcount = pd.DataFrame(r2_l.groupby([ 'parts','category']).size(), columns=['number'])
r2_lcount.reset_index(level=0, inplace=True)
r2_lcount.reset_index(level=0, inplace=True)
r2_l_user_group_by = r2_lcount.groupby(['parts','category']).agg({'number':'sum'})
r2_l_parts = r2_lcount.groupby(['parts']).agg({'number': 'sum'})
r2_l_percentage = r2_l_user_group_by.div(r2_l_parts, level='parts') * 100
r2_l_percentage.reset_index(level=0, inplace=True)
r2_l_percentage.to_csv('/home/verareyes/lp/street/analysis/r2_l_percentage.csv')




