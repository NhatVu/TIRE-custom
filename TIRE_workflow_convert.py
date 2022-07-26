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
import sys
import argparse


import utils
import TIRE
from importlib import reload 

# experiments 
# from experiments.EEG_L2 import EEG_L2_Experiment as X # original 
# from experiments.EEG_DMD import EEG_DMD_Experiment as X # Method 5: DMD, svd = 1 
# from experiments.EEG_DMD_L2 import EEG_DMD_L2_Experiment # Method 7: DMD, svd = 3, L2 norm
# from experiments.EEG_ICA_L2 import EEG_ICA_L2_Experiment as X # method 2
# from experiments.Original_L2 import Original_Experiment as X
# from experiments.Original_channel_0 import Original_Experiment as X
from experiments.Original_DMD import Original_Experiment as X 
# from experiments.EEG_ICA_DMD import EEG_ICA_DMD_Experiment as X

################################################
# setting env variable 
################################################3
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

# parsing input arguments
parser=argparse.ArgumentParser()

parser.add_argument('-t', '--type', type=str, help='Hyperparameter settings type: alpha or beta', required=True)
parser.add_argument('-d', '--datasetnumber', type=int, help='dataset number, from 1 to 8', required=True)

args=parser.parse_args()

hyper_type = args.type 
dataset_number = args.datasetnumber
print(f'hyper_type: {hyper_type}, dataset_number: {dataset_number}')
# %%
# ipynb, mỗi lần đổi code, phải restart kernal để load lại toàn bộ. Import thì chỉ lấy từ cache, ko lấy được code mới. Lý do

utils.setup_random_seed()

###############
workflow = X()
workflow.set_hyperparameter_type(hyper_type)
print(f'experiment name: {workflow.hyperparams.experiment_name}')

# ## Generate data

# training
# series = dataset_number
# print('training')
# training_timeseries, training_timeseries_len, training_windows_TD, training_windows_FD = workflow.get_timeseries(f'../data/eeg_grasp_and_lift/dataset{series}_training_data.csv')
# print('call breakpoint')
# training_breakpoints = workflow.get_breakpoint(training_timeseries_len, f'../data/eeg_grasp_and_lift/dataset{series}_training_label.csv')

# # validation
# print('validation')
# validation_timeseries, validation_timeseries_len, validation_windows_TD, validation_windows_FD = workflow.get_timeseries(f'../data/eeg_grasp_and_lift/dataset{series}_validation_data.csv')
# testing_breakpoints = workflow.get_breakpoint(validation_timeseries_len, f'../data/eeg_grasp_and_lift/dataset{series}_validation_label.csv')


# # testing
# print('testing')
# testing_timeseries, testing_timeseries_len, testing_windows_TD, testing_windows_FD = workflow.get_timeseries(f'../data/eeg_grasp_and_lift/dataset{series}_testing_data.csv')
# testing_breakpoints = workflow.get_breakpoint(testing_timeseries_len, f'../data/eeg_grasp_and_lift/dataset{series}_testing_label.csv')

print('training')
dataset_name = 'gauss-5-noise'
folder_prefix = f'../data-gen/{dataset_name}/{dataset_name}'
training_timeseries, training_timeseries_len, training_windows_TD, training_windows_FD = workflow.get_timeseries(f'{folder_prefix}-dataset{dataset_number}-training-data.csv')
print('call breakpoint')
training_breakpoints = workflow.get_breakpoint(training_timeseries_len, f'{folder_prefix}-dataset{dataset_number}-training-label.csv')

# validation
print('validation')
validation_timeseries, validation_timeseries_len, validation_windows_TD, validation_windows_FD = workflow.get_timeseries(f'{folder_prefix}-dataset{dataset_number}-validation-data.csv')
testing_breakpoints = workflow.get_breakpoint(validation_timeseries_len, f'{folder_prefix}-dataset{dataset_number}-validation-label.csv')


# testing
print('testing')
testing_timeseries, testing_timeseries_len, testing_windows_TD, testing_windows_FD = workflow.get_timeseries(f'{folder_prefix}-dataset{dataset_number}-testing-data.csv')
testing_breakpoints = workflow.get_breakpoint(testing_timeseries_len, f'{folder_prefix}-dataset{dataset_number}-testing-label.csv')



# ## Train the autoencoders
import timeit

start = timeit.default_timer()
shared_features_TD, shared_features_FD = workflow.train_autoencoder(training_windows_TD, training_windows_FD, validation_TD=validation_windows_TD, validation_FD = validation_windows_FD)

stop = timeit.default_timer()

print('Time training in minutes: ', (stop - start) / 60) 

# ## Postprocessing and peak detection
workflow.prepare_cal_metrics(dataset_number=dataset_number, dataset_name=dataset_name)



# print('AUC and F1 for training ')
# # predict shared features on testing data 
# training_shared_features_TD, training_shared_features_FD = workflow.predict(training_windows_TD, training_windows_FD)
# # post process for TD, FD and both, then save to file 
# workflow.dissimilarities_post_process(training_shared_features_TD, training_shared_features_FD)
# is_plot=False
# print('Get auc')
# f = open(workflow.metrics_path, 'a')
# f.write('Training score\n\n')
# f.close()
# workflow.get_auc(training_breakpoints, is_plot)

# reload(utils)
# print('get f1')
# f1s = workflow.get_f1(training_breakpoints, is_plot)



print('AUC and F1 for testing')
# predict shared features on testing data 
testing_shared_features_TD, testing_shared_features_FD = workflow.predict(testing_windows_TD, testing_windows_FD)
# post process for TD, FD and both, then save to file 
workflow.dissimilarities_post_process(testing_shared_features_TD, testing_shared_features_FD)
is_plot=False
print('Get auc')
f = open(workflow.metrics_path, 'a')
f.write('\nTesting score\n\n')
f.close()
workflow.get_auc(testing_breakpoints, is_plot)

reload(utils)
print('get f1')
f1s = workflow.get_f1(testing_breakpoints, is_plot)




