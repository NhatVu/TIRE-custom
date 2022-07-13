import pandas as pd
import numpy as np
import os

from experiments.hyperparameter import HyperParameter
import tensorflow as tf 
import utils
import TIRE
# from pydmd import DMD

class Experiment:
    def __init__(self) -> None:
        self.hyperparams = {}
    
    def set_hyperparameter_type(self, type:str):
        config = HyperParameter()
        config.domains = ['TD', 'FD', 'both']
        
        config.experiment_name = 'Unknown'
        config.type_setting = type
        config.tol_distances = [300]
        if type == 'alpha':
            config.window_size = 100
            #parameters TD
            # (8, 2), (16, 3)
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

    def set_experiment_name(self, name:str):
        self.hyperparams.experiment_name = name 

    def get_timeseries(self, file_path: str):
        pass 
    
    def get_breakpoint(self, timeseries_len:int, file_path: str):
        pass 

    def train_autoencoder(self, windows_TD, windows_FD, validation_TD=None, validation_FD=None):
        pass 

    def __predict_shared_features (self, windows, encoder, nr_ae, nr_shared):
        pass 

    def predict(self, windows_TD, windows_FD):
        pass 

    def dissimilarities_post_process(self, shared_features_TD, shared_features_FD):
        pass 

    def get_auc(self, breakpoints, is_plot=True):
        pass 

    def get_f1(self, breakpoints, is_plot=True):
        pass 

# Base class for One Dimension Experiment
class OneDimExperiment(Experiment):
    def __init__(self) -> None:
        super().__init__()
    
    def train_autoencoder(self, windows_TD, windows_FD, validation_TD=None, validation_FD=None):
    
        utils.setup_random_seed()
        shared_features_TD, encoder_TD = TIRE.train_AE(windows_TD, self.hyperparams.intermediate_dim_TD, self.hyperparams.latent_dim_TD, self.hyperparams.nr_shared_TD, self.hyperparams.nr_ae_TD, self.hyperparams.loss_weight_TD, nr_patience=5, nr_epochs=200, validation_data=validation_TD) 

        utils.setup_random_seed()
        shared_features_FD, encoder_FD = TIRE.train_AE(windows_FD, self.hyperparams.intermediate_dim_FD, self.hyperparams.latent_dim_FD, self.hyperparams.nr_shared_FD, self.hyperparams.nr_ae_FD, self.hyperparams.loss_weight_FD, nr_patience=5, nr_epochs=200, validation_data=validation_FD)

        self.encoder_TD = encoder_TD
        self.encoder_FD = encoder_FD

        # save model 
        dirname = os.path.dirname(__file__)
        save_folder_TD = os.path.join(dirname, f'../train/encoder_{self.hyperparams.experiment_name}_{self.hyperparams.type_setting}_TD')
        save_folder_FD = os.path.join(dirname, f'../train/encoder_{self.hyperparams.experiment_name}_{self.hyperparams.type_setting}_FD')

        encoder_TD.save_weights(save_folder_TD)
        encoder_FD.save_weights(save_folder_FD)
        return shared_features_TD, shared_features_FD 
    
    def __predict_shared_features (self, windows, encoder, nr_ae, nr_shared):
        new_windows = TIRE.prepare_input_paes(windows,nr_ae)
        encoded_windows_pae = encoder.predict(new_windows)
        encoded_windows = np.concatenate((encoded_windows_pae[:,0,:nr_shared],encoded_windows_pae[-nr_ae+1:,nr_ae-1,:nr_shared]),axis=0)
        return encoded_windows
    
    def predict(self, windows_TD, windows_FD):
        # if encoder doesn't exist in self. load it from file 
        if hasattr(self, 'encoder_TD') == False or hasattr(self, 'encoder_FD') == False:
            print('load encoder')
            hparam = self.hyperparams
            pae_TD, encoder_TD, decoder_TD = TIRE.create_parallel_aes(hparam.window_size, hparam.intermediate_dim_TD, hparam.latent_dim_TD,hparam.nr_ae_TD, hparam.nr_shared_TD, hparam.loss_weight_TD)

            pae_FD, encoder_FD, decoder_FD = TIRE.create_parallel_aes(hparam.nfft // 2 + 1, hparam.intermediate_dim_FD, hparam.latent_dim_FD,hparam.nr_ae_FD, hparam.nr_shared_FD, hparam.loss_weight_FD)

            dirname = os.path.dirname(__file__)
            save_folder_TD = os.path.join(dirname, f'../train/encoder_{self.hyperparams.experiment_name}_{self.hyperparams.type_setting}_TD')
            save_folder_FD = os.path.join(dirname, f'../train/encoder_{self.hyperparams.experiment_name}_{self.hyperparams.type_setting}_FD')
            
            encoder_TD.load_weights(save_folder_TD)
            encoder_FD.load_weights(save_folder_FD)

            self.encoder_TD = encoder_TD
            self.encoder_FD = encoder_FD

        encoded_windows_TD = self.__predict_shared_features(windows_TD, self.encoder_TD, self.hyperparams.nr_ae_TD, self.hyperparams.nr_shared_TD)
        encoded_windows_FD = self.__predict_shared_features(windows_FD, self.encoder_FD, self.hyperparams.nr_ae_FD, self.hyperparams.nr_shared_FD)

        return encoded_windows_TD, encoded_windows_FD

    def dissimilarities_post_process(self, shared_features_TD, shared_features_FD):
        dirname = os.path.dirname(__file__)
        for domain in self.hyperparams.domains:
            dissimilarities = TIRE.smoothened_dissimilarity_measures(shared_features_TD, shared_features_FD, domain, self.hyperparams.window_size)
            # save_folder = os.path.join(dirname, f'../data/dissimilarities_{self.hyperparams.experiment_name}_{self.hyperparams.type_setting}')
            # utils.create_folder_if_not_exist(save_folder)
            np.savetxt(f'{self.dissimilarity_folder}/dissimilarities_{domain}.txt', dissimilarities, fmt='%.20f')
            print(f'{self.dissimilarity_folder}/dissimilarities_{domain}.txt')
        
    def prepare_cal_metrics(self, dataset_number=0):
        dirname = os.path.dirname(__file__)
        utils.create_folder_if_not_exist(os.path.join(dirname, f'../metrics'))
        saving_path = os.path.join(dirname, f'../metrics/metrics_dataset{dataset_number}_{self.hyperparams.experiment_name}_{self.hyperparams.type_setting}.txt')
        if os.path.exists(saving_path):
            os.remove(saving_path)
        self.metrics_path = saving_path

        #create dissimilarity folder
        dis_folder = os.path.join(dirname, f'../data/dissimilarities_dataset{dataset_number}_{self.hyperparams.experiment_name}_{self.hyperparams.type_setting}')

        utils.create_folder_if_not_exist(dis_folder)
        self.dissimilarity_folder = dis_folder

    def get_auc(self, breakpoints, is_plot=True):
        dirname = os.path.dirname(__file__)
        for domain in self.hyperparams.domains:
            file_path = os.path.join(self.dissimilarity_folder, f'dissimilarities_{domain}.txt')
            dissimilarities = np.loadtxt(file_path)
            tol_distances = self.hyperparams.tol_distances
            auc = utils.get_auc(dissimilarities,tol_distances, breakpoints, is_plot)
            print(f'mode: {domain}, auc: {auc}')

            # save to file 
            utils.create_folder_if_not_exist(os.path.join(dirname, f'../metrics'))
            f = open(self.metrics_path, 'a')
            f.write(f'mode: {domain}, auc: {auc}\n')
            f.close()

    
    def get_f1(self, breakpoints, is_plot=True):
        dirname = os.path.dirname(__file__)
        for domain in self.hyperparams.domains:
            file_path = os.path.join(self.dissimilarity_folder, f'dissimilarities_{domain}.txt')
            dissimilarities = np.loadtxt(file_path)
            tol_distances = self.hyperparams.tol_distances
            f1s = utils.get_F1(dissimilarities,tol_distances, breakpoints, is_plot)
            f1max = []
            for i in range(len(tol_distances)):
                f1max.append(max(f1s[i]))
            print(f'mode: {domain}, f1 max: {f1max}')
            
            # save to file 
            f = open(self.metrics_path, 'a')
            f.write(f'mode: {domain}, f1 max: {f1max}\n')
            f.close()

# Base class for Each Dimension Experiment 
class EachDimExperiment(Experiment):
    pass 