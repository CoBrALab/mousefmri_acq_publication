#!/usr/bin/env python
# coding: utf-8

""" Purpose: some of these functions are copied from https://github.com/CoBrALab/MousePhgyMetrics (in order to detrend 
and smooth the pulseox and resp traces. Here, I want to plot heart beats, spo2 and resp all in the same panel and color-
code by phgy cluster.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import warnings
import os
import sys


######################################## FUNCTIONS #################################

def repeat_values_for_plotting(data_to_repeat, breaths_bool, breath_indices):
    '''For metrics where there is only 1 value per breath - duplicate the value until the next breath to create array
    with the length of all samples in original data - that way it can be plotted on the same graph as original trace.'''
    
    full_length_array = pd.Series(np.repeat(np.nan, len(breaths_bool)))
    
    for i in range(0, len(breath_indices)-1):
        full_length_array[breath_indices[i]:breath_indices[i+1]] = np.repeat(data_to_repeat[i], breath_indices[i+1]-breath_indices[i])
        
    return full_length_array        
        
def denoise_cardiac(pulseox_trace, sampling_rate, invert_bool):
    #convert the input array to a pandas Series, since the rolling window function is a pandas function
    raw_trace = pd.Series(pulseox_trace)
    
    #smooth the raw trace by taking the mean within a rolling window of 20 samples
    #using the Gaussian weighting makes every element in the smoothed series unique with higher precision
    trace_smoothed = raw_trace.rolling(20, center = True, min_periods = 1, win_type = 'gaussian').mean(std=5)
    
    if invert_bool == True:
        trace_smoothed = (-1)*trace_smoothed
    
    return trace_smoothed

def denoise_detrend_resp(raw_trace, sampling_rate, invert_bool):
    '''function to smooth and detrend the data'''
    #convert the input array to a pandas Series, since the rolling window function is a pandas function
    trace = pd.Series(raw_trace)
    
    #smooth the trace by taking the mean within a rolling window of 80 samples (Gaussian weighted mean, std =20)
    #using the Gaussian weighting makes every element in the smoothed series unique with higher precision
    trace_smoothed = raw_trace.rolling(40, center = True, min_periods = 1, win_type = 'gaussian').mean(std=10)
    
    #detrend the smoothed raw trace by subtracting the mean across 1 s (if necessary, invert the trace along y-axis first
    #inversion is to account for the fact that sometimes the resp pillow is placed backwards, so inspiration = down
    trend = trace_smoothed.rolling(sampling_rate, center = True, min_periods = 1).mean()
    if invert_bool:
        trace_smoothed_detrend_init = -1*trace_smoothed + trend
    else:
        trace_smoothed_detrend_init = trace_smoothed - trend
    trace_smoothed_detrend = trace_smoothed_detrend_init.reset_index(drop= True)
    
    return trace_smoothed_detrend

def downsample_to_once_per_sec(series_to_downsample, tot_num_samples, tot_length_seconds):
    sampling_rate = int(tot_num_samples/tot_length_seconds)
    series_reshaped = series_to_downsample.to_numpy().reshape((sampling_rate, tot_length_seconds), order = 'F')
    
    #if there are only NaNs in the entire second, will cause runtime warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        series_downsampled = np.nanmean(series_reshaped, axis=0)
    return pd.Series(series_downsampled)


