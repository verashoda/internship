#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:04:56 2019
Event and Presence of Types of Chatters
@author: verareyes
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

df = pd.read_csv('/home/verareyes/lp/street/database.csv')
count = pd.DataFrame(df.groupby(['event', 'category']).size(), columns=['number'])
count.reset_index(level=0, inplace=True)
count.reset_index(level=0, inplace=True)
count.reset_index(level=0, inplace=True)
user_group_by = count.groupby(['event','category']).agg({'number':'sum'})
event = count.groupby(['event']).agg({'number': 'sum'})
percentage = user_group_by.div(event, level='event') * 100

percentage.reset_index(level=0, inplace=True)
percentage.reset_index(level=0, inplace=True)

##### Top chat categories per event
break_event = percentage[(percentage.event == 'break') & (percentage.number > 10)]

before_match = percentage[(percentage.event == 'before match') & (percentage.number > 10)]

c_data = percentage[(percentage.event == 'character data') & (percentage.number > 10)]

final_round = percentage[(percentage.event == 'final round') & (percentage.number > 10)]

lose = percentage[(percentage.event == 'match lose') & (percentage.number > 10)]

round_one = percentage[(percentage.event == 'match round one') & (percentage.number > 10)]

round_two = percentage[(percentage.event == 'match round two') & (percentage.number > 10)]

win = percentage[(percentage.event == 'match win') & (percentage.number > 10)]

player_s = percentage[(percentage.event == 'player standings') & (percentage.number > 10)]

select_c = percentage[(percentage.event == 'select character') & (percentage.number > 10)]

story = percentage[(percentage.event == 'story') & (percentage.number > 10)]

training = percentage[(percentage.event == 'training') & (percentage.number > 10)]

trials = percentage[(percentage.event == 'trials') & (percentage.number > 10)]