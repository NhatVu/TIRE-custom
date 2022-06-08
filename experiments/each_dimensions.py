'''
1. get_settings(type='alpha' or 'beta')
2. get_timeseries()
3. get_breakpoint()
4. train_autoencoder()
5. dissimilarities_post_process
6. get_auc
7. get_f1
'''
from ctypes import util
import pandas as pd
import numpy as np
import os

from experiments.hyperparameter import HyperParameter

import utils
import TIRE

class Experiment:
    def __init__(self):
        self.hyperparams = {}
        
    
    def set_hyperparameter_type(self, type:str):
        config = HyperParameter()
        config.domains = ['TD', 'FD', 'both']
        config.experiment_name = 'each_dimensions'
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
        dirname = os.path.dirname(__file__)
        self.save_folder = os.path.join(dirname, f'../data/dissimilarities_{self.hyperparams.experiment_name}_{self.hyperparams.type_setting}')
        self.num_channels = 3

    def get_timeseries(self):
        # load hasc data 
        dirname = os.path.dirname(__file__)
        data_file = os.path.join(dirname, '../data/hasc-111018-165936-acc.csv')

        ts_df = pd.read_csv(data_file,header = None, names=['time', 'x', 'y', 'z'])
        timeseries = ts_df[['x', 'y', 'z']].to_numpy()
        timeseries = timeseries.T # change shape to (3, 39397)

        timeseries_len = len(timeseries[0])
        windows_TD = []
        windows_FD = []
        for series in timeseries:
            tmp_window_TD = utils.ts_to_windows(series, 0, self.hyperparams.window_size, 1)
            tmp_window_TD = utils.minmaxscale(tmp_window_TD,-1,1)
            tmp_windows_FD = utils.calc_fft(tmp_window_TD, self.hyperparams.nfft, self.hyperparams.norm_mode)

            windows_TD.append(tmp_window_TD)
            windows_FD.append(tmp_windows_FD)

        return timeseries, timeseries_len, windows_TD, windows_FD

    def get_breakpoint(self, timeseries_len:int):
        dirname = os.path.dirname(__file__)
        breakpoints_index_file = os.path.join(dirname, '../data/preprocess/hasc_label_index.txt')
        breakpoints_df = pd.read_csv(breakpoints_index_file, header=None)
        breakpoints_index = breakpoints_df[0].to_numpy()
        breakpoints_index = breakpoints_index - self.hyperparams.window_size # change index because we reduce the length of breakpoints 
        breakpoints = np.array([0] * (timeseries_len - 2* self.hyperparams.window_size + 1))

        breakpoints[breakpoints_index] = [1]*len(breakpoints_index)
        return breakpoints 

    def train_autoencoder(self, windows_TD, windows_FD):
        shared_features_TD = []
        shared_features_FD = []
        for i in range(self.num_channels):
            print(f'train AE for channel: {i}')
            tmp_shared_features_TD = TIRE.train_AE(windows_TD[i], self.hyperparams.intermediate_dim_TD, self.hyperparams.latent_dim_TD, self.hyperparams.nr_shared_TD, self.hyperparams.nr_ae_TD, self.hyperparams.loss_weight_TD, nr_patience=40)
            tmp_shared_features_FD = TIRE.train_AE(windows_FD[i], self.hyperparams.intermediate_dim_FD, self.hyperparams.latent_dim_FD, self.hyperparams.nr_shared_FD, self.hyperparams.nr_ae_FD, self.hyperparams.loss_weight_FD, nr_patience=40)

            shared_features_TD.append(tmp_shared_features_TD)
            shared_features_FD.append(tmp_shared_features_FD)

        return shared_features_TD, shared_features_FD 

    def dissimilarities_post_process(self, shared_features_TD, shared_features_FD):     
        for domain in ['TD', 'FD', 'both']:
            dissimilarities = []
            for i in range(self.num_channels):
                tmp_dissimilarities = TIRE.smoothened_dissimilarity_measures(shared_features_TD[i], shared_features_FD[i], domain, self.hyperparams.window_size)
                dissimilarities.append(tmp_dissimilarities)
            dissimilarities = np.array(dissimilarities)
            # save 

            utils.create_folder_if_not_exist(self.save_folder)
            np.savetxt(f'{self.save_folder}/dissimilarities_{domain}.txt', dissimilarities, fmt='%.20f')
        

    def get_auc(self, breakpoints):
        for domain in self.hyperparams.domains:
            file_path = os.path.join(self.save_folder, f'dissimilarities_{domain}.txt')
            dissimilarities = np.loadtxt(file_path)
            prominence = utils.create_prominence_from_multi_channels(self.num_channels, dissimilarities, self.hyperparams.window_size)
            tol_distances = [300]
            print(f'mode: {domain}')
            auc = utils.get_auc_v2(prominence,tol_distances, breakpoints)
    
    def get_f1(self, breakpoints):
        pass
        # for domain in self.hyperparams.domains:
        #     file_path = os.path.join(self.save_folder, f'dissimilarities_{domain}.txt')
        #     dissimilarities = np.loadtxt(file_path)
        #     tol_distances = [300]
        #     print(f'mode: {domain}')
        #     auc = utils.get_F1(dissimilarities,tol_distances, breakpoints)



