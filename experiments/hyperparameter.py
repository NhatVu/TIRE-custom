class HyperParameter:
    def __init__(self) -> None:
        self.domains = ['TD', 'FD', 'both']
        self.experiment_name = 'replication'
        self.window_size = 100
        self.tol_distances = [15]
        #parameters TD
        self.intermediate_dim_TD=0
        self.latent_dim_TD=1 #h^TD in paper
        self.nr_shared_TD=1 #s^TD in paper
        self.K_TD = 4 #as in paper
        self.nr_ae_TD= self.K_TD+1 #number of parallel AEs = K+1
        self.loss_weight_TD=1 #lambda_TD in paper

        #parameters FD
        self.intermediate_dim_FD=0
        self.latent_dim_FD=1 #h^FD in paper
        self.nr_shared_FD=1 #s^FD in paper
        self.K_FD = 2 #as in paper
        self.nr_ae_FD=self.K_FD+1 #number of parallel AEs = K+1
        self.loss_weight_FD=1 #lambda^FD in paper
        self.nfft = 30 #number of points for DFT
        self.norm_mode = "timeseries" #for calculation of DFT, should the timeseries have mean zero or each window?
        self.type_setting = 'alpha'
    
