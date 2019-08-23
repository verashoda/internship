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
###### count the type of chat per event and percentage
req = req_count.groupby(['event']).agg({'number': 'sum'})
req_percentage = req_count.div(req, level='event') * 100
##### count the message lenth and percentage


### COMMENTERS
comment = groups[(groups.category == 'comment') & (groups.number > 25)]
commenters = comment['user'].tolist()
commenters_data = df[df.user.isin(commenters)]
com_count = pd.DataFrame(commenters_data.groupby(['event', 'category']).size(), columns=['number'])
##### message length
commenters_data['message_len'] = commenters_data['message'].astype(str).apply(len)
commenters_data['polarity'] = commenters_data['message'].map(lambda text: TextBlob(text).sentiment.polarity) 
###### count the type of chat per event and percentage
com = com_count.groupby(['event']).agg({'number': 'sum'})
com_percentage = com_count.div(com, level='event') * 100



#### CHARACTER FANS
character1 = groups[(groups.category == 'cinfo') & (groups.number > 25)]
character2 = groups[(groups.category == 'caction') & (groups.number > 25)]
character3 = groups[(groups.category == 'cdesign') & (groups.number > 25)]
characters = character1['user'].tolist() + character2['user'].tolist() + character3['user'].tolist()
characters = list(set(characters))
characters_data = df[df.user.isin(characters)]
character_count = pd.DataFrame(characters_data.groupby(['event', 'category']).size(), columns=['number'])
##### message length
characters_data['message_len'] = characters_data['message'].astype(str).apply(len) 
characters_data['polarity'] = characters_data['message'].map(lambda text: TextBlob(text).sentiment.polarity) 
###### count the type of chat per event and percentage
char = character_count.groupby(['event']).agg({'number': 'sum'})
char_percentage = character_count.div(char, level='event') * 100



#### PRAISERS
praise = groups[(groups.category == 'goodp') & (groups.number > 25)]
praisers = praise['user'].tolist()
praisers_data = df[df.user.isin(praisers)]
praise_count = pd.DataFrame(commenters_data.groupby(['event', 'category']).size(), columns=['number'])
##### message length
praisers_data['message_len'] = praisers_data['message'].astype(str).apply(len)
praisers_data['polarity'] = praisers_data['message'].map(lambda text: TextBlob(text).sentiment.polarity) 
###### count the type of chat per event and percentage
pra = praise_count.groupby(['event']).agg({'number': 'sum'})
praise_percentage = praise_count.div(pra, level='event') * 100


#### BOOERS
boo = groups[(groups.category == 'badp') & (groups.number > 25)]
booers = boo['user'].tolist()
booers_data = df[df.user.isin(booers)]
boo_count = pd.DataFrame(booers_data.groupby(['event', 'category']).size(), columns=['number'])
##### message length
booers_data['message_len'] = booers_data['message'].astype(str).apply(len)
booers_data['polarity'] = booers_data['message'].map(lambda text: TextBlob(text).sentiment.polarity) 
##### count the type of chat per event and percentage
booer = boo_count.groupby(['event']).agg({'number': 'sum'})
boo_percentage = boo_count.div(booer, level='event') * 100


#### GAME FANS
game = groups[(groups.category == 'game') & (groups.number > 25)]
gamers = game['user'].tolist()
gamers_data = df[df.user.isin(gamers)]
gamers_count = pd.DataFrame(gamers_data.groupby(['event', 'category']).size(), columns=['number'])
##### message length
gamers_data['message_len'] = gamers_data['message'].astype(str).apply(len)
gamers_data['polarity'] = gamers_data['message'].map(lambda text: TextBlob(text).sentiment.polarity) 
###### count the type of chat per event and percentage
gam = gamers_count.groupby(['event']).agg({'number': 'sum'})
game_percentage = gamers_count.div(gam, level='event') * 100





