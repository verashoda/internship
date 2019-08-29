#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:16:00 2019
Personalities Visualization
@author: verareyes
"""
##### Import libraries
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import plotly.io as pio
pio.renderers.default = "browser"


df = pd.read_csv('/home/verareyes/lp/street/analysis/user_personalities/requester.csv')
df = df.drop(['category'], axis=1)
value = [df[x].values.tolist() for x in df.columns]

        

##### Visualizing Data
top_labels = ['bad','action','design','info','com','game','good move','other','quest','req',]
colors = ['rgb(214, 12, 140)','rgb(10, 140, 208)','rgb(10, 140, 208)','rgb(10, 140, 208)','rgb(222, 223, 0)','rgb(100, 0, 10)','rgb(255, 144, 14)','rgb(7,40,89)','rgb(255, 65, 54)','rgb(12, 102, 14)'] 
x_data = value
y_data = ['before match','break','character data','final round','match lose','round one','round two','match win','player standings','select character','story','training','trials']
fig = go.Figure()
for i in range(0, len(x_data[0])):
    for xd, yd in zip(x_data, y_data):
        fig.add_trace(go.Bar(
            x=[xd[i]], y=[yd],
            orientation='h',
            marker=dict(
                color=colors[i],
                line=dict(color='rgb(248, 248, 249)', width=1)
            )
        ))

fig.update_layout(
    xaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
        domain=[0.15, 1]
    ),
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
    ),
    barmode='stack',
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    margin=dict(l=10, r=10, t=100, b=10),
    showlegend=False,
)

annotations = []

for yd, xd in zip(y_data, x_data):
    # labeling the y-axis
    annotations.append(dict(xref='paper', yref='y',
                            x=0.14, y=yd,
                            xanchor='right',
                            text=str(yd),
                            font=dict(family='Arial', size=10,
                                      color='rgb(67, 67, 67)'),
                            showarrow=False, align='right'))
    # labeling the first percentage of each bar (x_axis)
    annotations.append(dict(xref='x', yref='y',
                            x=xd[0]/2, y=yd,
                            text=str(xd[0]) + '%',
                            font=dict(family='Arial', size=10,
                                      color='rgb(248, 248, 255)'),
                            showarrow=False))
    # labeling the first Likert scale (on the top)
    if yd == y_data[-1]:
        annotations.append(dict(xref='x', yref='paper',
                                x=xd[0]/2, y=1.1,
                                text=top_labels[0],
                                font=dict(family='Arial', size=10,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False))
    space = xd[0]
    for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=space + (xd[i]/2), y=yd,
                                    text=str(xd[i]) + '%',
                                    font=dict(family='Arial', size=10,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
            # labeling the Likert scale
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i]/2), y=1.1,
                                        text=top_labels[i],
                                        font=dict(family='Arial', size=10,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += xd[i]

fig.update_layout(annotations=annotations)

fig.show()
