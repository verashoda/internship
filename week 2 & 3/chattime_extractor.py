#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:42:12 2019
Unique values across colums with Pandas
@author: verareyes
"""

#import pandas library
import pandas as pd

#reading the csv file
df = pd.read_csv('/home/verareyes/twitch_clips/fortnite/fort_01_startendtime.csv' )

#getting unique values across the csv
df1 = pd.unique(df[['Start_time', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 'End_time']].values.ravel('K'))

#transforming output to data frame
dataFrame = pd.DataFrame(df1)
dataFrame.columns = ['Unique_times']

#exporting data to csv file
dataFrame.to_csv('/home/verareyes/twitch_clips/fortnite/fort_01_uniquetimes.csv', index=False, header=True)



