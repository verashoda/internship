#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:50:06 2019

@author: verareyes
"""
import pandas as pd

#getting the frequency sum per 10 seconds
df = pd.read_csv('/home/verareyes/twitch_clips/fortnite/fort_01_freq.csv')
df['cum_sum'] = df.Frequency.rolling(window=5).sum()
##print(df)
#
df.to_csv('/home/verareyes/twitch_clips/fortnite/fort_01_freqsum.csv', index=False, header=True)

#getting the maximum values
#count = df.groupby(['cum_sum'])

