# %% [markdown]
# # CPD using TIRE

# %%
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Lambda, Input, Dense
# from tensorflow.keras.models import Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import os


import utils
import TIRE
from importlib import reload 

# experiments 
# import experiments.EEG_L2 as x # original 
# import experiments.EEG_DMD as x # Method 5: DMD, svd = 1 
import experiments.EEG_DMD_L2 as x # Method 7: DMD, svd = 3, L2 norm

# setting env variable 
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

# %%
# ipynb, mỗi lần đổi code, phải restart kernal để load lại toàn bộ. Import thì chỉ lấy từ cache, ko lấy được code mới. Lý do

# %%
workflow = x.Experiment()
workflow.set_hyperparameter_type('alpha')
workflow.hyperparams.experiment_name

# %% [markdown]
# ## Generate data

# %%
# training
series = 1
print('training')
training_timeseries, training_timeseries_len, training_windows_TD, training_windows_FD = workflow.get_timeseries(f'../data/eeg_grasp_and_lift/dataset{series}_training_data.csv')
print('call breakpoint')
training_breakpoints = workflow.get_breakpoint(training_timeseries_len, f'../data/eeg_grasp_and_lift/dataset{series}_training_label.csv')

# validation
print('validation')
validation_timeseries, validation_timeseries_len, validation_windows_TD, validation_windows_FD = workflow.get_timeseries(f'../data/eeg_grasp_and_lift/dataset{series}_validation_data.csv')
testing_breakpoints = workflow.get_breakpoint(validation_timeseries_len, f'../data/eeg_grasp_and_lift/dataset{series}_validation_label.csv')


# testing
print('testing')
testing_timeseries, testing_timeseries_len, testing_windows_TD, testing_windows_FD = workflow.get_timeseries(f'../data/eeg_grasp_and_lift/dataset{series}_testing_data.csv')
testing_breakpoints = workflow.get_breakpoint(testing_timeseries_len, f'../data/eeg_grasp_and_lift/dataset{series}_testing_label.csv')


# training
# print('training')
# training_timeseries, training_timeseries_len, training_windows_TD, training_windows_FD = workflow.get_timeseries('../../Data/grasp-and-lift-eeg-detection/train/subj10_series1_data.csv')
# training_breakpoints = workflow.get_breakpoint(training_timeseries_len, '../../Data/grasp-and-lift-eeg-detection/train/subj10_series1_events.csv')

# # testing
# print('testing')
# testing_timeseries, testing_timeseries_len, testing_windows_TD, testing_windows_FD = workflow.get_timeseries('../../Data/grasp-and-lift-eeg-detection/train/subj10_series1_data.csv')
# testing_breakpoints = workflow.get_breakpoint(testing_timeseries_len, '../../Data/grasp-and-lift-eeg-detection/train/subj10_series1_events.csv')

# %% [markdown]
# ## Train the autoencoders
import timeit

start = timeit.default_timer()
# shared_features_TD, shared_features_FD = workflow.train_autoencoder(training_windows_TD, training_windows_FD, validation_TD=validation_windows_TD, validation_FD = validation_windows_FD)

stop = timeit.default_timer()

print('Time training in minutes: ', (stop - start) / 60) 

# %% [markdown]
# ## Postprocessing and peak detection

# %%
# predict shared features on testing data 
testing_shared_features_TD, testing_shared_features_FD = workflow.predict(testing_windows_TD, testing_windows_FD)
# post process for TD, FD and both, then save to file 
workflow.dissimilarities_post_process(testing_shared_features_TD, testing_shared_features_FD)

is_plot=False
# %%
print('Get auc')
workflow.get_auc(testing_breakpoints, is_plot)

# %%
reload(utils)
print('get f1')
f1s = workflow.get_f1(testing_breakpoints, is_plot)




