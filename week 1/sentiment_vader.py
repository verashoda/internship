# -*- coding: utf-8 -*-
"""
Spyder Editor

V. Shoda
07.24.2019
"""

#Importing libraries
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

#open file and sentiment analysis per line
with open("/home/verareyes/twitch_clips/owl/text analysis/raw text/OWL10_raw.txt") as fp:
    scores =[]
    sentences = fp.readline()
    cnt = 1
    while sentences:
        score = analyzer.polarity_scores(sentences)
        scores.append(score)
        sentences = fp.readline()
        cnt += 1

#formatting the output as dataframe
dataFrame = pd.DataFrame(scores)
print(dataFrame)
dataFrame.mean()

#writing output to csv file
dataFrame.to_csv('/home/verareyes/twitch_clips/owl/text analysis/raw text/OWL10_sent.csv', index=True, header=True)




    


        

