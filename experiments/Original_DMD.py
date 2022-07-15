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

class Original_Experiment(OneDimExperiment):
    def __init__(self):
        super().__init__()
    
    def set_hyperparameter_type(self, type:str):
        super().set_hyperparameter_type(type)
        self.hyperparams.experiment_name = 'original_dmd'
        
    def get_timeseries(self, file_path = '../data/eeg_subj1_series1_data.csv'):
        # load hasc data 
        dirname = os.path.dirname(__file__)
        data_file = os.path.join(dirname, file_path)

        signal_df = pd.read_csv(data_file)
        print(f'signal_df shape: {signal_df.shape}')
        

        timeseries = signal_df.to_numpy()
        dmd = DMD(svd_rank=1)
        dmd.fit(timeseries)
        timeseries = dmd.modes.T[0].real

        windows_TD = utils.ts_to_windows(timeseries, 0, self.hyperparams.window_size, 1)
        windows_TD = utils.minmaxscale(windows_TD,-1,1)
        print(f'timeseries shape: {timeseries.shape}, windows_TD shape: {windows_TD.shape}')


        # for windows_FD
        # signal_df = pd.read_csv(data_file)
        timeseries_tmp = timeseries
        tmp_TD = utils.ts_to_windows(timeseries_tmp, 0, self.hyperparams.window_size, 1)
        tmp_TD = utils.minmaxscale(tmp_TD,-1,1)
        windows_FD = utils.calc_fft(tmp_TD, self.hyperparams.nfft, self.hyperparams.norm_mode)

        return timeseries, len(timeseries), windows_TD, windows_FD

    def get_breakpoint(self, timeseries_len:int, file_path = '../data/eeg_subj1_series1_events.csv'):
        dirname = os.path.dirname(__file__)
        breakpoints_index_file = os.path.join(dirname, file_path)

        labels_df = pd.read_csv(breakpoints_index_file)

        result = labels_df['col_0'].to_numpy()

        return result[self.hyperparams.window_size: len(result) - self.hyperparams.window_size + 1]



