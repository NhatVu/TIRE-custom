'''
1. get_settings(type='alpha' or 'beta')
2. get_timeseries()
3. get_breakpoint()
4. train_autoencoder()
5. dissimilarities_post_process
6. get_auc
7. get_f1
'''
import pandas as pd
import numpy as np
import os

from experiments.BaseExperiments import OneDimExperiment

import utils

class EEG_L2_Experiment(OneDimExperiment):
    def __init__(self):
        super().__init__()
    
    def set_hyperparameter_type(self, type:str):
        super().set_hyperparameter_type(type)
        print(f'hyperparameter: {self.hyperparams}')
        self.hyperparams.experiment_name = 'eeg_l2'
        
    def get_timeseries(self, file_path = '../data/eeg_subj1_series1_data.csv'):
        # load hasc data 
        dirname = os.path.dirname(__file__)
        data_file = os.path.join(dirname, file_path)

        egg_signal_df = pd.read_csv(data_file)
        egg_signal_df.drop(['id'], axis=1, inplace=True)

        timeseries = np.sqrt(np.square(egg_signal_df).sum(axis=1)).to_numpy()
        print(f'timeseries shape: {timeseries.shape}')

        windows_TD = utils.ts_to_windows(timeseries, 0, self.hyperparams.window_size, 1)
        windows_TD = utils.minmaxscale(windows_TD,-1,1)
        windows_FD = utils.calc_fft(windows_TD, self.hyperparams.nfft, self.hyperparams.norm_mode)

        return timeseries, len(timeseries), windows_TD, windows_FD

    def get_breakpoint(self, timeseries_len:int, file_path = '../data/eeg_subj1_series1_events.csv'):
        dirname = os.path.dirname(__file__)
        breakpoints_index_file = os.path.join(dirname, file_path)

        labels_df = pd.read_csv(breakpoints_index_file)
        labels_df.drop(['id'], axis=1, inplace=True)

        change_event_index = labels_df.max(axis=1).to_numpy()
        possitive_index = np.where(np.array(change_event_index) > 0)[0]
        change_index = []
        i = 0
        while i < len(possitive_index) - 1:
            change_index.append(possitive_index[i])
            while i + 1 < len(possitive_index) and possitive_index[i] + 1 == possitive_index[i + 1]: 
                i += 1
            if i + 1 < len(possitive_index):
                change_index.append(possitive_index[i] + 1)
            i += 1
        result = np.array([0] * len(change_event_index))
        result[change_index] = [1] * len(change_index)

        return result[self.hyperparams.window_size: len(change_event_index) - self.hyperparams.window_size + 1]



