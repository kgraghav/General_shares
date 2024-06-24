# -*- coding: utf-8 -*-
"""
Created on Wed Jul 6 13:48:29 2022

@author: raghakg
"""

#%%
import numpy as np
import pandas as pd
import math as mth
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy import *
import seaborn as sns
from numpy import array
from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -sf')

plt.close("all") 

filenames=['',]
write_filenames=['']
edge_colors=['blue','red']
n_rows=len(filenames)
n_cols=1# duration of the video
# fig=plt.figure()
# matplot subplot

for j,filename,write_filename in zip(range(len(filenames)),filenames,write_filenames):
    print(filename)
    data=pd.read_csv(filename)
    unique_dates=data['EVENT_DATE'].drop_duplicates()[:5].tolist()
    # matplot subplot
    fig, ax = plt.subplots()
    plt.bar(x=1,height=1)
    duration = len(unique_dates)
    frames=np.array([])
        # method to get frames
    def make_frame(t):
        frame_time=np.arange(duration)
        for i,d in zip(range(1,len(frame_time)),range(len(unique_dates))):
            date=unique_dates[d]
            t_start=frame_time[i-1]
            t_stop=frame_time[i]
            print(date)
            plot_var=data[data['EVENT_DATE']==date].DEF_PRESS.value_counts(
            normalize=True)
            if t_start<=t<=t_stop:
                # clear
                ax.clear()
                # returning numpy im
                plt.bar(x=plot_var.index,height=plot_var)
                plt.title(date)
                print('FRAME DETAILS: ',t,t_start,t_stop)
                # plt.pause(t)
                # np.append(frames,fig)
                # print(frames)
            else:
                continue;
        return mplfig_to_npimage(fig)

    # creating animation
    animation = VideoClip(make_frame, duration = duration)
    
    # displaying clip
    animation .ipython_display(fps = 20, loop = False, autoplay = False)
    
    # Saving animation
    animation.write_videofile(write_filename+'.mp4',fps=30,
    codec='libx264')
    
    # animation.ipython_display(width = 480,fps=30)
