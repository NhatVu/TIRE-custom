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
import experiments.utils_eeg as utils_eeg


import utils
import TIRE
from pydmd import DMD

class Experiment:
    def __init__(self):
        self.hyperparams = {}
    
    def set_hyperparameter_type(self, type:str):
        config = HyperParameter()
        config.domains = ['TD', 'FD', 'both']
        config.experiment_name = 'eeg_dmd_l2'
        config.type_setting = type
        if type == 'alpha':
            config.window_size = 100
            #parameters TD
            config.intermediate_dim_TD=0
            config.latent_dim_TD=1 #h^TD in paper
            config.nr_shared_TD=1 #s^TD in paper
            config.K_TD = 2 #as in paper
            config.nr_ae_TD= config.K_TD+1 #number of parallel AEs = K+1
            config.loss_weight_TD=1 #lambda_TD in paper

            #parameters FD
            config.intermediate_dim_FD=0
            config.latent_dim_FD=1 #h^FD in paper
            config.nr_shared_FD=1 #s^FD in paper
            config.K_FD = 2 #as in paper
            config.nr_ae_FD=config.K_FD+1 #number of parallel AEs = K+1
            config.loss_weight_FD=1 #lambda^FD in paper
            config.nfft = 30 #number of points for DFT
            config.norm_mode = "timeseries" #for calculation of DFT, should the timeseries have mean zero or each window?
        elif type == 'beta':
            config.window_size = 100
            #parameters TD
            config.intermediate_dim_TD=0
            config.latent_dim_TD=3 #h^TD in paper
            config.nr_shared_TD=2 #s^TD in paper
            config.K_TD = 2 #as in paper
            config.nr_ae_TD= config.K_TD+1 #number of parallel AEs = K+1
            config.loss_weight_TD=1 #lambda_TD in paper

            #parameters FD
            config.intermediate_dim_FD=10
            config.latent_dim_FD=1 #h^FD in paper
            config.nr_shared_FD=1 #s^FD in paper
            config.K_FD = 2 #as in paper
            config.nr_ae_FD=config.K_FD+1 #number of parallel AEs = K+1
            config.loss_weight_FD=1 #lambda^FD in paper
            config.nfft = 30 #number of points for DFT
            config.norm_mode = "timeseries" #for calculation of DFT, should the timeseries have mean zero or each window?
        else:
            print('Please input alpha or beta')
        
        self.hyperparams = config


    def get_timeseries(self, file_path='../data/eeg_subj1_series1_data.csv'):
        # load hasc data 
        dirname = os.path.dirname(__file__)
        data_file = os.path.join(dirname, file_path)

        egg_signal_df = pd.read_csv(data_file)
        egg_signal_df.drop(['id'], axis=1, inplace=True)

        dmd = DMD(svd_rank=3)
        dmd.fit(egg_signal_df.to_numpy())

        timeseries = dmd.modes.T.real

        timeseries = np.sqrt(np.square(timeseries).sum(axis=0))
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

        result = utils_eeg.create_break_point_index(labels_df=labels_df)
        return result[self.hyperparams.window_size: len(result) - self.hyperparams.window_size + 1]

    def train_autoencoder(self, windows_TD, windows_FD, validation_TD=None, validation_FD = None):
        shared_features_TD, encoder_TD = TIRE.train_AE(windows_TD, self.hyperparams.intermediate_dim_TD, self.hyperparams.latent_dim_TD, self.hyperparams.nr_shared_TD, self.hyperparams.nr_ae_TD, self.hyperparams.loss_weight_TD, nr_patience=20, nr_epochs=200)
        shared_features_FD, encoder_FD = TIRE.train_AE(windows_FD, self.hyperparams.intermediate_dim_FD, self.hyperparams.latent_dim_FD, self.hyperparams.nr_shared_FD, self.hyperparams.nr_ae_FD, self.hyperparams.loss_weight_FD, nr_patience=20, nr_epochs=200)

        self.encoder_TD = encoder_TD
        self.encoder_FD = encoder_FD
        return shared_features_TD, shared_features_FD 
    
    def __predict_shared_features (self, windows, encoder, nr_ae, nr_shared):
        new_windows = TIRE.prepare_input_paes(windows,nr_ae)
        encoded_windows_pae = encoder.predict(new_windows)
        encoded_windows = np.concatenate((encoded_windows_pae[:,0,:nr_shared],encoded_windows_pae[-nr_ae+1:,nr_ae-1,:nr_shared]),axis=0)
        return encoded_windows
    
    def predict(self, windows_TD, windows_FD):
        encoded_windows_TD = self.__predict_shared_features(windows_TD, self.encoder_TD, self.hyperparams.nr_ae_TD, self.hyperparams.nr_shared_TD)
        encoded_windows_FD = self.__predict_shared_features(windows_FD, self.encoder_FD, self.hyperparams.nr_ae_FD, self.hyperparams.nr_shared_FD)

        return encoded_windows_TD, encoded_windows_FD

    def dissimilarities_post_process(self, shared_features_TD, shared_features_FD):
        dirname = os.path.dirname(__file__)
        for domain in self.hyperparams.domains:
            dissimilarities = TIRE.smoothened_dissimilarity_measures(shared_features_TD, shared_features_FD, domain, self.hyperparams.window_size)
            save_folder = os.path.join(dirname, f'../data/dissimilarities_{self.hyperparams.experiment_name}_{self.hyperparams.type_setting}')
            utils.create_folder_if_not_exist(save_folder)
            np.savetxt(f'{save_folder}/dissimilarities_{domain}.txt', dissimilarities, fmt='%.20f')
            print(f'{save_folder}/dissimilarities_{domain}.txt')
        

    def get_auc(self, breakpoints, is_plot=True):
        print(f'Change')
        dirname = os.path.dirname(__file__)
        for domain in self.hyperparams.domains:
            file_path = os.path.join(dirname, f'../data/dissimilarities_{self.hyperparams.experiment_name}_{self.hyperparams.type_setting}/dissimilarities_{domain}.txt')
            dissimilarities = np.loadtxt(file_path)
            tol_distances = [100, 200, 300]
            print(f'mode: {domain}')
            auc = utils.get_auc(dissimilarities,tol_distances, breakpoints, is_plot)
    
    def get_f1(self, breakpoints, is_plot=True):
        dirname = os.path.dirname(__file__)
        for domain in self.hyperparams.domains:
            file_path = os.path.join(dirname, f'../data/dissimilarities_{self.hyperparams.experiment_name}_{self.hyperparams.type_setting}/dissimilarities_{domain}.txt')
            dissimilarities = np.loadtxt(file_path)
            tol_distances = [100, 200, 300]
            f1s = utils.get_F1(dissimilarities,tol_distances, breakpoints, is_plot)
            





