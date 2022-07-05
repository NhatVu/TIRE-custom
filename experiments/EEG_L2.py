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
import experiments.utils_eeg as utils_eeg


import utils
from pydmd import DMD

class EEG_L2_Experiment(OneDimExperiment):
    def __init__(self):
        super().__init__()
    
    def set_hyperparameter_type(self, type:str):
        super().set_hyperparameter_type(type)
        self.hyperparams.experiment_name = 'eeg_l2'
        
    def get_timeseries(self, file_path = '../data/eeg_subj1_series1_data.csv'):
        # load hasc data 
        dirname = os.path.dirname(__file__)
        data_file = os.path.join(dirname, file_path)

        egg_signal_df = pd.read_csv(data_file)
        egg_signal_df.drop(['id'], axis=1, inplace=True)
        

        # timeseries = np.sqrt(np.square(egg_signal_df).sum(axis=1)).to_numpy()
        timeseries = egg_signal_df.to_numpy()
        dmd = DMD(svd_rank=3)
        dmd.fit(timeseries)
        timeseries = dmd.modes
        print(f'timeseries shape: {timeseries.shape}')
        windows_TD = utils.ts_to_windows(timeseries, 0, self.hyperparams.window_size, 1)
        windows_TD = utils.minmaxscale(windows_TD,-1,1)

        # for windows_FD
        timeseries_TD = np.sqrt(np.square(egg_signal_df).sum(axis=1)).to_numpy()
        tmp_TD = utils.ts_to_windows(timeseries_TD, 0, self.hyperparams.window_size, 1)
        tmp_TD = utils.minmaxscale(tmp_TD,-1,1)
        windows_FD = utils.calc_fft(tmp_TD, self.hyperparams.nfft, self.hyperparams.norm_mode)

        return timeseries, len(timeseries), windows_TD, windows_FD

    def get_breakpoint(self, timeseries_len:int, file_path = '../data/eeg_subj1_series1_events.csv'):
        dirname = os.path.dirname(__file__)
        breakpoints_index_file = os.path.join(dirname, file_path)

        labels_df = pd.read_csv(breakpoints_index_file)
        labels_df.drop(['id'], axis=1, inplace=True)

        result = utils_eeg.create_break_point_index(labels_df=labels_df)

        return result[self.hyperparams.window_size: len(result) - self.hyperparams.window_size + 1]



