#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:22:16 2019
Histogram with pandas dataframe
@author: verareyes
"""

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

df = pd.read_csv('/home/verareyes/twitch_clips/fortnite/fort_01_freqsum.csv')
df1 = pd.DataFrame(df)
cum_sum = df1['cum_sum']

#creating the count labels per bin
your_bins=15
data=cum_sum
arr=plt.hist(data, color=['orange'],bins=your_bins)
for i in range(your_bins):
    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))
#plt.hist([cum_sum], color=['orange'])
plt.xlabel("score")
plt.ylabel("observations")
plt.savefig('/home/verareyes/twitch_clips/fortnite/fort_01_timehist.png')
plt.show()


#CREATING THE HISTOGRAM

#ax = df1.hist(column='cum_sum', bins=16, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
#ax = ax[0]
#
##Despite
#x.spines['right'].set_visible(False)
#x.spines['top'].set_visible(False)
#x.spines['left'].set_visible(False)
#
##Switch off ticks
#x.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='on', left='off', right='off', labelleft='on')
#
##draw horizontal axis lines
#vals = x.get_yticks()
#for tick in vals:
#    x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
#
##Remove title
#x.set_title("")
#
##Set x-axis label
#x.set_xlabel("Sum of chat entries", labelpad=20, weight='bold', size=12)
#
##Set y axis label
#x.set_ylabel("Observations", labelpad=20, weight='bold', size=12)
#
##Format y-axis label
#x.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))

#plt.savefig('/home/verareyes/histogram_OWL01.png')

