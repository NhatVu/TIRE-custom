#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import warnings
import time, copy

import utils
import TIRE
import simulate
import math


# ## Set parameters

# #### alpha settings

window_size = 100
domain = "both" #choose from: TD (time domain), FD (frequency domain) or both

#parameters TD
intermediate_dim_TD=0
latent_dim_TD=1 #h^TD in paper
nr_shared_TD=1 #s^TD in paper
K_TD = 2 #as in paper
nr_ae_TD= K_TD+1 #number of parallel AEs = K+1
loss_weight_TD=1 #lambda_TD in paper

#parameters FD
intermediate_dim_FD=0
latent_dim_FD=1 #h^FD in paper
nr_shared_FD=1 #s^FD in paper
K_FD = 2 #as in paper
nr_ae_FD=K_FD+1 #number of parallel AEs = K+1
loss_weight_FD=1 #lambda^FD in paper
nfft = 30 #number of points for DFT
norm_mode = "timeseries" #for calculation of DFT, should the timeseries have mean zero or each window?


# #### beta settings
# 

# window_size = 100
# domain = "both" #choose from: TD (time domain), FD (frequency domain) or both

# #parameters TD
# intermediate_dim_TD=0
# latent_dim_TD=3 #h^TD in paper
# nr_shared_TD=2 #s^TD in paper
# K_TD = 2 #as in paper
# nr_ae_TD= K_TD+1 #number of parallel AEs = K+1
# loss_weight_TD=1 #lambda_TD in paper

# #parameters FD
# intermediate_dim_FD=10
# latent_dim_FD=1 #h^FD in paper
# nr_shared_FD=1 #s^FD in paper
# K_FD = 2 #as in paper
# nr_ae_FD=K_FD+1 #number of parallel AEs = K+1
# loss_weight_FD=1 #lambda^FD in paper
# nfft = 30 #number of points for DFT
# norm_mode = "timeseries" #for calculation of DFT, should the timeseries have mean zero or each window?


# load hasc data 
# data_file = './data/preprocess/hasc_l2_norm.csv'
data_file = './data/hasc-111018-165936-acc.csv'
breakpoints_index_file = './data/preprocess/hasc_label_index.txt'
num_channels = 3

ts_df = pd.read_csv(data_file,header = None, names=['time', 'x', 'y', 'z'])
timeseries = ts_df[['x', 'y', 'z']].to_numpy()
timeseries = timeseries.T # change shape to (3, 39397)

breakpoints_df = pd.read_csv(breakpoints_index_file, header=None)
breakpoints_index = breakpoints_df[0].to_numpy()
breakpoints_index = breakpoints_index - window_size # change index because we reduce the length of breakpoints 

'''
1. Generate time series, 
2. convert time series into window_TD
3. Convert time serites into window_FD
4. Create groundtruth
'''
timeseries_len = len(timeseries[0])
windows_TD = []
windows_FD = []
for series in timeseries:
    tmp_window_TD = utils.ts_to_windows(series, 0, window_size, 1)
    tmp_window_TD = utils.minmaxscale(tmp_window_TD,-1,1)
    tmp_windows_FD = utils.calc_fft(tmp_window_TD, nfft, norm_mode)

    windows_TD.append(tmp_window_TD)
    windows_FD.append(tmp_windows_FD)

# len(breakpoints) = len(timeseries) - 2*window_size + 1
# breakpoints_index = [179]
print(f'len: {len(breakpoints_index)}, value: {breakpoints_index}')
breakpoints = np.array([0] * (timeseries_len - 2*window_size + 1))

breakpoints[breakpoints_index] = [1]*len(breakpoints_index)
print(timeseries_len, len(breakpoints))


# ## Train the autoencoders

shared_features_TD = []
shared_features_FD = []
for i in range(num_channels):
    print(f'train AE for channel: {i}')
    tmp_shared_features_TD = TIRE.train_AE(windows_TD[i], intermediate_dim_TD, latent_dim_TD, nr_shared_TD, nr_ae_TD, loss_weight_TD, nr_patience=40)
    tmp_shared_features_FD = TIRE.train_AE(windows_FD[i], intermediate_dim_FD, latent_dim_FD, nr_shared_FD, nr_ae_FD, loss_weight_FD, nr_patience=40)

    shared_features_TD.append(tmp_shared_features_TD)
    shared_features_FD.append(tmp_shared_features_FD)


# ## Postprocessing and peak detection


#we calculate the smoothened dissimilarity measure and the corresponding prominence-based change point scores
print(f'calculate smoothened dissimilarity')
for domain in ['TD', 'FD', 'both']:
    dissimilarities = []
    for i in range(num_channels):
        tmp_dissimilarities = TIRE.smoothened_dissimilarity_measures(shared_features_TD[i], shared_features_FD[i], domain, window_size)
        dissimilarities.append(tmp_dissimilarities)
    dissimilarities = np.array(dissimilarities)
    # save 
    np.savetxt(f'./data/dissimilarities/dissimilarities_{domain}.txt', dissimilarities, fmt='%f')

