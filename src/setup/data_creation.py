# %% Import Standard Packages
import os
import numpy as np

# %% Start Main Class

class Dataset:
    """
    Class for building the datasets (Domain, collocation and exact(with noise) ):
    Input:
        - par, a param object that store all the parameters

    Objects:
        - pde_type
        - name_example = name of dataset we want to build/load
        - prop_exact = proportion of domain data to use as exact data
        - prop_collocation = proportion of domain data to use as collocation data
        - n_input = 1,2 or 3 (1D, 2D or 3D experiment)
        - noise_lv = noise level in exact data

    Private:
        - __flag_dataset_build = flag to build/load the dataset just one time
        - __flag_dataset_noise = flag to add the noise to exact data just one time
    """

    def __init__(self, par):
        """The constructor"""
        self.pde_type = par.pde
        self.name_example = par.dataset # name of dataset to use

        self.__num_fitting = par.experiment["num_fitting"] # num of exact data
        self.__num_collocation = par.experiment["num_collocation"] # num of collocation data
        self.noise_lv = par.experiment["noise_lv"]
        
        self.n_input   = par.phys_dim.n_input
        self.n_out_par = par.phys_dim.n_out_par
        self.n_out_sol = par.phys_dim.n_out_sol

        self.__flag_dataset_build = False
        self.__flag_dataset_noise = False
        np.random.seed(par.utils["random_seed"])

    @property
    def dom_data(self):
        self.build_dataset()
        return self.inputs_dom, self.U_dom, self.F_dom

    @property
    def coll_data(self):
        """Return collocation data"""
        self.build_dataset()
        return self.inputs_coll, self.U_coll, self.F_coll

    @property
    def exact_data(self):
        """ return exact data"""
        self.build_dataset()
        return self.inputs_exact, self.U_exact, self.F_exact

    @property
    def exact_data_noise(self):
        """ return exact data + noise """
        self.build_dataset()
        self.build_noisy_dataset()
        return self.inputs_exact, self.U_with_noise, self.F_with_noise

    @property
    def num_collocation(self):
        """get number of collocation points"""
        return self.__num_collocation

    @property
    def num_fitting(self):
        """get number of exact points"""
        return self.__num_fitting

    def __load_dataset(self):
        """load data from dataset"""
        path = os.path.join("../data", self.name_example)

        inputs = list()
        for var_file in os.listdir(path):
            if not var_file[-4:] == ".npy": continue
            if var_file[:-4] in ["x","y","z"]:
                inputs.append(np.load(os.path.join(path,var_file))[...,None])
        inputs = np.concatenate(inputs, axis=-1)
        inputs = inputs.astype(np.float32)

        u = np.load(os.path.join(path,"u.npy")).astype(np.float32)
        f = np.load(os.path.join(path,"f.npy")).astype(np.float32)

        if(len(f.shape)==1): u = u[...,None] # from shape (n_coll, ) -> (n_coll, 1)
        if(len(f.shape)==1): f = f[...,None] # add the last dimension only if we are in 1D case

        # store domain datasets
        self.inputs_dom = inputs
        self.U_dom = u
        self.F_dom = f

        # n_domain is simply the length of one of the dataset, x for instance
        index = list(range(inputs.shape[0]))

        # build collocation dataset from domain dataset
        # np.random.shuffle(index) USE ONLY FOR LINSPACE!
        if self.num_collocation > inputs.shape[0] : 
            raise Exception(f'Num collocation cannot be bigger than dataset size: {self.num_collocation} > {inputs.shape[0]}')
        index_coll = index[:self.num_collocation]
        self.inputs_coll = inputs[index_coll,:]
        self.U_coll = u[index_coll,:]
        self.F_coll = f[index_coll,:]

        # build exact dataset from domain dataset
        # np.random.shuffle(index) USE ONLY FOR LINSPACE!
        if self.num_fitting > inputs.shape[0] : 
            raise Exception(f'Num fitting cannot be bigger than dataset size: {self.num_fitting} > {inputs.shape[0]}')
        index_exac = index[:self.num_fitting]
        self.inputs_exact = inputs[index_exac,:]
        self.U_exact = u[index_exac,:]
        self.F_exact = f[index_exac,:]


    def build_dataset(self):
        """
        Build dataset:
        call the functions to build the dataset
        """
        if not self.__flag_dataset_build:  # we build the dataset only the first time
            self.__load_dataset()
            self.__flag_dataset_build = True

    def build_noisy_dataset(self):
        """
        Add noise to exact data
        """
        self.build_dataset()
        if self.__flag_dataset_noise: return  # we add the noise only the first time
        
        u_error = np.random.normal(0, self.noise_lv, self.U_exact.shape).astype("float32")
        self.U_with_noise = self.U_exact + u_error

        # not so useful... we use onlt U_with_noise, not F 
        f_error = np.random.normal(0, self.noise_lv, self.F_exact.shape).astype("float32")
        self.F_with_noise = self.F_exact + f_error

        self.__flag_dataset_noise = True
