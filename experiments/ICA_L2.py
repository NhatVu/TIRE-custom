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
from sklearn.decomposition import FastICA

from experiments.hyperparameter import HyperParameter
from experiments.BaseExperiments import OneDimExperiment


import utils
import TIRE

class ICA_L2_Experiment(OneDimExperiment):
    def __init__(self):
        super().__init__()
    
    def set_hyperparameter_type(self, type:str):
        super().set_hyperparameter_type(type)
        self.hyperparams.experiment_name = 'ICA_L2'


    def get_timeseries(self, file_path = '../data/eeg_subj1_series1_data.csv'):
        # load hasc data 
        dirname = os.path.dirname(__file__)
        data_file = os.path.join(dirname, file_path)

        signal_df = pd.read_csv(data_file)
        print(f'signal_df shape: {signal_df.shape}')
        timeseries = signal_df.to_numpy() # have to use copy(). It's seem the internal np array has changed somewhere
        

        transformer = FastICA(n_components=None,
                random_state=0,
                whiten='unit-variance')
        timeseries = transformer.fit_transform(timeseries)
        timeseries = np.sqrt(np.square(timeseries).sum(axis=1))

        windows_TD = utils.ts_to_windows(timeseries, 0, self.hyperparams.window_size, 1)
        windows_TD = utils.minmaxscale(windows_TD,-1,1)
        print(f'timeseries shape: {timeseries.shape}, windows_TD shape: {windows_TD.shape}')


        # for windows_FD
        # timeseries_tmp = np.sqrt(np.square(signal_df).sum(axis=1)).to_numpy()
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



