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

from experiments.hyperparameter import HyperParameter
from experiments.BaseExperiments import OneDimExperiment

import experiments.utils_eeg as utils_eeg

import utils
import TIRE
from pydmd import DMD

class EEG_ICA_DMD_Experiment(OneDimExperiment):
    def __init__(self):
        super().__init__()
    
    def set_hyperparameter_type(self, type:str):
        super().set_hyperparameter_type(type)
        self.hyperparams.experiment_name = 'EEG_ICA_DMD'


    def get_timeseries(self, file_path = '../data/eeg_subj1_series1_data.csv'):
        # load hasc data 
        dirname = os.path.dirname(__file__)
        data_file = os.path.join(dirname, file_path)

        egg_signal_df = pd.read_csv(data_file)
        egg_signal_df.drop(['id'], axis=1, inplace=True)

        from sklearn.decomposition import FastICA
        transformer = FastICA(n_components=None,
                        random_state=0,
                        whiten='unit-variance')
        ica_signal = transformer.fit_transform(egg_signal_df.to_numpy())

        dmd = DMD(svd_rank=1)
        dmd.fit(ica_signal)

        timeseries = dmd.modes.T[0].real

        # timeseries = np.sqrt(np.square(timeseries).sum(axis=0))
        print(f'timeseries shape: {timeseries.shape}')

        windows_TD = utils.ts_to_windows(timeseries, 0, self.hyperparams.window_size, 1)
        windows_TD = utils.minmaxscale(windows_TD,-1,1)
        windows_FD = utils.calc_fft(windows_TD, self.hyperparams.nfft, self.hyperparams.norm_mode)

        return timeseries, len(timeseries), windows_TD, windows_FD

    def get_breakpoint(self, timeseries_len:int, file_path):
        dirname = os.path.dirname(__file__)
        breakpoints_index_file = os.path.join(dirname, file_path)

        labels_df = pd.read_csv(breakpoints_index_file) # , 'FirstDigitTouch', 'LiftOff', 'BothReleased'
        labels_df.drop(['id'], axis=1, inplace=True)

        result = utils_eeg.create_break_point_index(labels_df=labels_df)
        return result[self.hyperparams.window_size: len(result) - self.hyperparams.window_size + 1]
            





