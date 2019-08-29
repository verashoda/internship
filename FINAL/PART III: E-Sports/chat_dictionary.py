#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:41:56 2019
Grouping the time and chat data to a dictionary
@author: verareyes
"""
import pandas as pd

df = pd.read_csv('/home/verareyes/twitch_clips/fortnite/fort_01.csv')
 #covert to data frame
df1 = pd.DataFrame(df)
# group by column
grouped = df1.groupby('time')

#viewing the contents for each group
#for name, group in grouped:
#    print (name)
#    print (group)
#setting the groups as dictionary keys
#owl_dict0 = dict(tuple(df1.groupby('time')))
#owl_dict = df.set_index('time').transpose().to_dict(orient='list')
owl_dict = df1.groupby('time')['chat'].apply(lambda g: g.values.tolist()).to_dict()

#dataframe = pd.DataFrame.from_dict(owl_dict, orient='index')
#dataframe['Unique_times']=dataframe.index
#print(dataframe)

#"""Part 2""""
#Get the corresponding values of keys in Unique_times

rawdata = pd.read_csv('/home/verareyes/twitch_clips/fortnite/fort_01_uniquetimes.csv')
unique_t = pd.DataFrame(rawdata)
unique_t = unique_t['Unique_times'].str[1:]

my_data = pd.DataFrame(unique_t, columns = ['Unique_times'])
#
##look up values of unique_times in owl_dictionary

my_data['chat'] = my_data['Unique_times'].map(owl_dict)
#
##export output dataframe into csv
#
my_data.to_csv('/home/verareyes/twitch_clips/fortnite/fort_01_excitedchat.csv', index=False, header=True)

#new_data = unique_t.append(dataframe, ignore_index=True)
#
#final_data = new_data.groupby('Unique_times')
#final_data.sum().reset_index().to_csv('/home/verareyes/output.csv')





















