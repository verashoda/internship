#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:17:34 2019
Testing the Chat Personalities of Top 100 Chatters
@author: verareyes
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

df = pd.read_csv('/home/verareyes/lp/street/database.csv')
user_count = pd.DataFrame(df['user'].value_counts())
#user_count['name'] = user_count.index
#users = pd.DataFrame(user_count)
#users.reset_index(level=0, inplace=True)
#users = users.drop(['index'], axis=1)
#users = users.rename(columns={'user':'counts', 'name':'user'})
#top_users = users[:105]
#low_users = users[800:]
#del_users = low_users['user'].tolist()
#user_data = df
#user_data = user_data[~user_data.user.isin(del_users)]
#count = pd.DataFrame(user_data.groupby(['user', 'category']).size(), columns=['number'])
#count['group'] = count.index
#count.reset_index(level=0, inplace=True)
#count.reset_index(level=0, inplace=True)
#count.reset_index(level=0, inplace=True)
#count = count.drop(['group'], axis=1)
#user_group_by = count.groupby(['user','category']).agg({'number':'sum'})
#user = count.groupby(['user']).agg({'number': 'sum'})
#percentage = user_group_by.div(user, level='user') * 100

##### breakdown into groups
#groups = percentage
#groups.reset_index(level=0, inplace=True)
#groups.reset_index(level=0, inplace=True)
#groups.reset_index(level=0, inplace=True)
#groups['number'].plot(kind='hist')

##### Finding types of chatters

### REQUESTERS
#request = groups[(groups.category == 'request') & (groups.number > 25)]
#requesters = request['user'].tolist()
#requesters_data = df[df.user.isin(requesters)]
#req_count = pd.DataFrame(requesters_data.groupby(['user', 'event']).size(), columns=['number'])

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

###### bi-grams
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

##### tri-grams
#def get_top_trigrams(corpus, n=None):
#    vec = CountVectorizer(ngram_range=(3,3), stop_words = 'english').fit(corpus)
#    bag_of_words = vec.transform(corpus)
#    sum_words = bag_of_words.sum(axis=0)
#    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#    return words_freq[:n]
#
#common_words = get_top_trigrams(requesters_data['message'], 20) #change input with the dataframe name
##for word, freq in common_words:
##    print(word, freq)
#df1 = pd.DataFrame(common_words, columns = ['message', 'count'])
#df1.groupby('message').sum()['count'].sort_values(ascending=False).plot(kind='bar')



### Getting Type of Chatters Top Events
#req_count.groupby(['user','event']).agg({'number':'sum'})
#req_user = req_count.groupby(['user']).agg({'number': 'sum'})
#percent_req = req_count.div(user, level='user') * 100
#percent_req.reset_index(level=0, inplace=True)
#percent_req.reset_index(level=0, inplace=True)
#req_events = percent_req[(percent_req.number > 10)]
#req_events = req_events.drop(['user'], axis=1)
###### Requesters Type of Events Percentage
#req_event_count = pd.DataFrame(req_events.groupby(['event']).size(), columns=['number'])
#req_event_count.groupby(['event']).agg({'number':'sum'})
#req_events = req_event_count.groupby(['event']).agg({'number': 'sum'})
#req_event_count['percentage'] = req_event_count['number']/req_event_count['number'].sum()

##### COMMENTERS
#comments = groups[(groups.category == 'comment') & (groups.number > 25)]
#commenters = comments['user'].tolist()
#commenters_data = df[df.user.isin(commenters)]
#com_count = pd.DataFrame(commenters_data.groupby(['user', 'event']).size(), columns=['number'])
#
###### Getting Type of Chatters Top Events
#com_count.groupby(['user','event']).agg({'number':'sum'})
#com_user = com_count.groupby(['user']).agg({'number': 'sum'})
#percent_com = com_count.div(user, level='user') * 100
#percent_com.reset_index(level=0, inplace=True)
#percent_com.reset_index(level=0, inplace=True)
#com_events = percent_com[(percent_com.number > 10)]
#com_events = com_events.drop(['user'], axis=1)
###### Requesters Type of Events Percentage
#com_event_count = pd.DataFrame(com_events.groupby(['event']).size(), columns=['number'])
#com_event_count.groupby(['event']).agg({'number':'sum'})
#com_events = com_event_count.groupby(['event']).agg({'number': 'sum'})
#com_event_count['percentage'] = com_event_count['number']/com_event_count['number'].sum()


##### CHARACTER ACTION FANS
#caction = groups[(groups.category == 'caction') & (groups.number > 25)]
#cactioners = caction['user'].tolist()
#cactioners_data = df[df.user.isin(cactioners)]
#cac_count = pd.DataFrame(cactioners_data.groupby(['user', 'event']).size(), columns=['number'])
#
###### Getting Type of Chatters Top Events
#cac_count.groupby(['user','event']).agg({'number':'sum'})
#cac_user = cac_count.groupby(['user']).agg({'number': 'sum'})
#percent_cac = cac_count.div(user, level='user') * 100
#percent_cac.reset_index(level=0, inplace=True)
#percent_cac.reset_index(level=0, inplace=True)
#cac_events = percent_cac[(percent_cac.number > 10)]
#cac_events = cac_events.drop(['user'], axis=1)
###### Requesters Type of Events Percentage
#cac_event_count = pd.DataFrame(cac_events.groupby(['event']).size(), columns=['number'])
#cac_event_count.groupby(['event']).agg({'number':'sum'})
#cac_events = cac_event_count.groupby(['event']).agg({'number': 'sum'})
#cac_event_count['percentage'] = cac_event_count['number']/cac_event_count['number'].sum()

##### CHARACTER DESIGN FANS
#cdesign = groups[(groups.category == 'cdesign') & (groups.number > 25)]
#cdesigners = cdesign['user'].tolist()
#cdesigners_data = df[df.user.isin(cdesigners)]
#cde_count = pd.DataFrame(cdesigners_data.groupby(['user', 'event']).size(), columns=['number'])
#
###### Getting Type of Chatters Top Events
#cde_count.groupby(['user','event']).agg({'number':'sum'})
#cde_user = cde_count.groupby(['user']).agg({'number': 'sum'})
#percent_cde = cde_count.div(user, level='user') * 100
#percent_cde.reset_index(level=0, inplace=True)
#percent_cde.reset_index(level=0, inplace=True)
#cde_events = percent_cde[(percent_cde.number > 10)]
#cde_events = cde_events.drop(['user'], axis=1)
###### Requesters Type of Events Percentage
#cde_event_count = pd.DataFrame(cde_events.groupby(['event']).size(), columns=['number'])
#cde_event_count.groupby(['event']).agg({'number':'sum'})
#cde_events = cde_event_count.groupby(['event']).agg({'number': 'sum'})
#cde_event_count['percentage'] = cde_event_count['number']/cde_event_count['number'].sum()

##### CHARACTER INFORMATION / FEELERS
#cinfo = groups[(groups.category == 'cinfo') & (groups.number > 25)]
#cinformers = cinfo['user'].tolist()
#cinformers_data = df[df.user.isin(cinformers)]
#cin_count = pd.DataFrame(cinformers_data.groupby(['user', 'event']).size(), columns=['number'])
#
###### Getting Type of Chatters Top Events
#cin_count.groupby(['user','event']).agg({'number':'sum'})
#cin_user = cin_count.groupby(['user']).agg({'number': 'sum'})
#percent_cin = cin_count.div(user, level='user') * 100
#percent_cin.reset_index(level=0, inplace=True)
#percent_cin.reset_index(level=0, inplace=True)
#cin_events = percent_cin[(percent_cin.number > 10)]
#cin_events = cin_events.drop(['user'], axis=1)
###### Requesters Type of Events Percentage
#cin_event_count = pd.DataFrame(cin_events.groupby(['event']).size(), columns=['number'])
#cin_event_count.groupby(['event']).agg({'number':'sum'})
#cin_events = cin_event_count.groupby(['event']).agg({'number': 'sum'})
#cin_event_count['percentage'] = cin_event_count['number']/cin_event_count['number'].sum()



##### PRAISERS
#goodp = groups[(groups.category == 'goodp') & (groups.number > 25)]
#gooders = goodp['user'].tolist()
#gooders_data = df[df.user.isin(gooders)]
#good_count = pd.DataFrame(gooders_data.groupby(['user', 'event']).size(), columns=['number'])
#
###### Getting Type of Chatters Top Events
#good_count.groupby(['user','event']).agg({'number':'sum'})
#good_user = good_count.groupby(['user']).agg({'number': 'sum'})
#percent_good = good_count.div(user, level='user') * 100
#percent_good.reset_index(level=0, inplace=True)
#percent_good.reset_index(level=0, inplace=True)
#good_events = percent_good[(percent_good.number > 10)]
#good_events = good_events.drop(['user'], axis=1)
###### Requesters Type of Events Percentage
#good_event_count = pd.DataFrame(good_events.groupby(['event']).size(), columns=['number'])
#good_event_count.groupby(['event']).agg({'number':'sum'})
#good_events = good_event_count.groupby(['event']).agg({'number': 'sum'})
#good_event_count['percentage'] = good_event_count['number']/good_event_count['number'].sum()
#


##### BOOERS
#badp = groups[(groups.category == 'badp') & (groups.number > 25)]
#booers = badp['user'].tolist()
#booers_data = df[df.user.isin(booers)]
#bad_count = pd.DataFrame(booers_data.groupby(['user', 'event']).size(), columns=['number'])
#
###### Getting Type of Chatters Top Events
#bad_count.groupby(['user','event']).agg({'number':'sum'})
#bad_user = bad_count.groupby(['user']).agg({'number': 'sum'})
#percent_bad = bad_count.div(user, level='user') * 100
#percent_bad.reset_index(level=0, inplace=True)
#percent_bad.reset_index(level=0, inplace=True)
#bad_events = percent_bad[(percent_bad.number > 10)]
#bad_events = bad_events.drop(['user'], axis=1)
###### Requesters Type of Events Percentage
#bad_event_count = pd.DataFrame(bad_events.groupby(['event']).size(), columns=['number'])
#bad_event_count.groupby(['event']).agg({'number':'sum'})
#bad_events = bad_event_count.groupby(['event']).agg({'number': 'sum'})
#bad_event_count['percentage'] = bad_event_count['number']/bad_event_count['number'].sum()



##### QUESTIONERS
#question = groups[(groups.category == 'question') & (groups.number > 25)]
#questioners = question['user'].tolist()
#questioners_data = df[df.user.isin(questioners)]
#question_count = pd.DataFrame(questioners_data.groupby(['user', 'event']).size(), columns=['number'])
#
###### Getting Type of Chatters Top Events
#question_count.groupby(['user','event']).agg({'number':'sum'})
#question_user = question_count.groupby(['user']).agg({'number': 'sum'})
#percent_question = question_count.div(user, level='user') * 100
#percent_question.reset_index(level=0, inplace=True)
#percent_question.reset_index(level=0, inplace=True)
#question_events = percent_question[(percent_question.number > 10)]
#question_events = question_events.drop(['user'], axis=1)
###### Requesters Type of Events Percentage
#question_event_count = pd.DataFrame(question_events.groupby(['event']).size(), columns=['number'])
#question_event_count.groupby(['event']).agg({'number':'sum'})
#question_events = question_event_count.groupby(['event']).agg({'number': 'sum'})
#question_event_count['percentage'] = question_event_count['number']/question_event_count['number'].sum()



##### GAME INFO MASTERS
#game = groups[(groups.category == 'game') & (groups.number > 25)]
#gamers = game['user'].tolist()
#gamers_data = df[df.user.isin(gamers)]
#game_count = pd.DataFrame(gamers_data.groupby(['user', 'event']).size(), columns=['number'])
#
###### Getting Type of Chatters Top Events
#game_count.groupby(['user','event']).agg({'number':'sum'})
#game_user = game_count.groupby(['user']).agg({'number': 'sum'})
#percent_game = game_count.div(user, level='user') * 100
#percent_game.reset_index(level=0, inplace=True)
#percent_game.reset_index(level=0, inplace=True)
#game_events = percent_game[(percent_game.number > 10)]
#game_events = game_events.drop(['user'], axis=1)
###### Requesters Type of Events Percentage
#game_event_count = pd.DataFrame(game_events.groupby(['event']).size(), columns=['number'])
#game_event_count.groupby(['event']).agg({'number':'sum'})
#game_events = game_event_count.groupby(['event']).agg({'number': 'sum'})
#game_event_count['percentage'] = game_event_count['number']/game_event_count['number'].sum()



##### NOISE MAKERS
#other = groups[(groups.category == 'other') & (groups.number > 25)]
#others = other['user'].tolist()
#others_data = df[df.user.isin(others)]
#other_count = pd.DataFrame(others_data.groupby(['user', 'event']).size(), columns=['number'])

##### Getting Type of Chatters Top Events
#other_count.groupby(['user','event']).agg({'number':'sum'})
#other_user = other_count.groupby(['user']).agg({'number': 'sum'})
#percent_other = other_count.div(user, level='user') * 100
#percent_other.reset_index(level=0, inplace=True)
#percent_other.reset_index(level=0, inplace=True)
#other_events = percent_other[(percent_other.number > 10)]
#other_events = other_events.drop(['user'], axis=1)
##### Requesters Type of Events Percentage
#other_event_count = pd.DataFrame(other_events.groupby(['event']).size(), columns=['number'])
#other_event_count.groupby(['event']).agg({'number':'sum'})
#other_events = other_event_count.groupby(['event']).agg({'number': 'sum'})
#other_event_count['percentage'] = other_event_count['number']/other_event_count['number'].sum()





