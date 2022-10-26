import os
import numpy as np
import warnings

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
        
        self.pde_type = par.pde
        self.problem  = par.problem
        self.name_example = par.folder_name
        self.mesh_type = par.config.analytical_domain["mesh_type"]

        self.num_fitting     = par.experiment["num_fitting"]
        self.num_collocation = par.experiment["num_collocation"]
        self.noise_lv = par.experiment["noise_lv"]
        
        self.n_input   = par.phys_dim.n_input
        self.n_out_par = par.phys_dim.n_out_par
        self.n_out_sol = par.phys_dim.n_out_sol

        self.__flag_dataset_build = False
        self.__flag_dataset_noise = False
        np.random.seed(par.utils["random_seed"])

        self.build_dataset()
        self.build_noisy_dataset()

    @property
    def dom_data(self):
        return self.inputs_dom, self.U_dom, self.F_dom
    
    @dom_data.setter
    def dom_data(self, *values):
        self.inputs_dom = values[0]
        self.U_dom = values[1]
        self.F_dom = values[2]

    @property
    def coll_data(self):
        return self.inputs_coll, self.U_coll, self.F_coll

    @coll_data.setter
    def coll_data(self, *values):
        self.inputs_coll = values[0]
        self.U_coll = values[1]
        self.F_coll = values[2]

    @property
    def exact_data(self):
        return self.inputs_exact, self.U_exact, self.F_exact
    
    @exact_data.setter
    def exact_data(self, *values):
        self.inputs_exact = values[0]
        self.U_exact = values[1]
        self.F_exact = values[2]

    @property
    def noise_data(self):
        return self.inputs_exact, self.U_noise, self.F_noise

    @noise_data.setter
    def noise_data(self, *values):
        self.inputs_exact = values[0]
        self.U_noise = values[1]
        self.F_noise = values[2]

    def __load_dataset(self):
        """load data from dataset"""
        path = os.path.join("../data", self.problem)
        path = os.path.join(path, self.name_example)

        inputs = list()
        for var_file in os.listdir(path):
            if not var_file[-4:] == ".npy": continue
            if var_file[:-4] in ["x","y","z"]:
                inputs.append(np.load(os.path.join(path,var_file))[...,None])
        inputs = np.concatenate(inputs, axis=-1)
        inputs = inputs.astype(np.float32)

        u = np.load(os.path.join(path,"u.npy")).astype(np.float32)
        f = np.load(os.path.join(path,"f.npy")).astype(np.float32)

        if(len(u.shape)==1): u = u[...,None] # from shape (n_coll, ) -> (n_coll, 1)
        if(len(f.shape)==1): f = f[...,None] # add the last dimension only if we are in 1D case

        # store domain datasets
        self.inputs_dom = inputs
        self.U_dom = u
        self.F_dom = f

        # build collocation dataset from domain dataset
        if self.num_collocation > inputs.shape[0] : 
            raise Exception(f'Num collocation cannot be bigger than dataset size: {self.num_collocation} > {inputs.shape[0]}')
        index_coll = self.__select_indexes(inputs.shape[0], self.num_collocation)
        self.inputs_coll = inputs[index_coll,:]
        self.U_coll = u[index_coll,:]
        self.F_coll = f[index_coll,:]

        # build exact dataset from domain dataset
        if self.num_fitting > inputs.shape[0] : 
            raise Exception(f'Num fitting cannot be bigger than dataset size: {self.num_fitting} > {inputs.shape[0]}')
        index_exac = self.__select_indexes(inputs.shape[0], self.num_fitting)
        self.inputs_exact = inputs[index_exac,:]
        self.U_exact = u[index_exac,:]
        self.F_exact = f[index_exac,:]

    def __select_indexes(self, res, num):
        # n_domain is simply the length of one of the dataset, x for instance
        index = list(range(res))
        match self.mesh_type:
            case "random"  : return index[:num]
            case "sobol"   :
                if not ((num-1) & (num-2)): return index[:num] + [index[-1]]
                elif not ((num) & (num-1)): return index[:num]
                elif num < 80: 
                    warnings.warn("Non optimal choice of resolution for Sobol mesh")
                    return index[:num]
            case "uniform" : 
                delta = (res - res//num * num) // 2 
                new_index = index[delta::res//num]
                new_index[0], new_index[-1] = index[0], index[-1]
                return new_index
            case _ : Exception("This mesh type doesn't exists")


    def build_dataset(self):
        """ call the functions to build the dataset """
        if not self.__flag_dataset_build:
            self.__load_dataset()
            self.__flag_dataset_build = True

    def build_noisy_dataset(self):
        """ Add noise to exact data """
        self.build_dataset()
        if self.__flag_dataset_noise: return
        
        u_error = np.random.normal(0, self.noise_lv, self.U_exact.shape).astype("float32")
        self.U_noise = self.U_exact + u_error

        f_error = np.random.normal(0, self.noise_lv, self.F_exact.shape).astype("float32")
        self.F_noise = self.F_exact + f_error

        self.__flag_dataset_noise = True
