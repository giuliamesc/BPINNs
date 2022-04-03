# %% Import Standard Packages
import os
import numpy as np

# %% Start Main Class

class dataset_class:
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
        - _flag_dataset_build = flag to build/load the dataset just one time
        - _flag_dataset_noise = flag to add the noise to exact data just one time

        (only if analytical)
            - n_domain = number of domain points
    """

    def __init__(self,par):
        """The constructor"""
        self.pde_type = par.pde
        self.name_example = par.experiment["dataset"]    # name of dataset to use

        self.prop_exact = par.experiment["prop_exact"] # prop of exact data
        self.prop_collocation = par.experiment["prop_collocation"] # prop of exact data
        self.n_input = par.n_input # 1D,2D,3D

        self.n_out_par = par.n_out_par
        self.n_out_sol = par.n_out_sol
        self.noise_lv = par.experiment["noise_lv"]

        self._flag_dataset_build = False
        self._flag_dataset_noise = False
        np.random.seed(par.utils["random_seed"])

        self.n_domain = 0
        self.n_collocation = 0
        self.n_exact = 0

# %% Main Functions

    def _load_dataset(self):
        """load data from dataset"""
        path = os.path.join("../data",self.name_example)

        if(self.n_input in [1,2,3]):
            x = np.load(os.path.join(path,"x.npy"))[...,None]
            inputs = x
            if(self.n_input > 1):
                y = np.load(os.path.join(path,"y.npy"))[...,None]
                inputs = np.concatenate((inputs,y),axis=1)
            if(self.n_input == 3):
                z = np.load(os.path.join(path,"z.npy"))[...,None]
                inputs = np.concatenate((inputs,z),axis=1)
            inputs = inputs.astype(np.float32)

            u = np.load(os.path.join(path,"u.npy")).astype(np.float32)
            f = np.load(os.path.join(path,"f.npy")).astype(np.float32)

            if(len(f.shape)==1): u = u[...,None] # from shape (n_coll, ) -> (n_coll, 1)
            if(len(f.shape)==1): f = f[...,None] # add the last dimension only if we are in 1D case

            # store domain datasets
            self.inputs_dom = inputs
            self.U_dom = u
            self.F_dom = f

            #n_domain is simply the length of one of the dataset, x for instance
            self.n_domain = len(x)
            index = list(range(self.n_domain))


            # compute n_collocation using n_domain and prop_collocation (min 10)
            self.n_collocation = max(int(self.n_domain*self.prop_collocation),10)
            # build collocation dataset from domain dataset
            np.random.shuffle(index)
            index_coll = index[:self.n_collocation]
            self.inputs_coll = inputs[index_coll,:]
            self.U_coll = u[index_coll,:]
            self.F_coll = f[index_coll,:]

            # compute n_exact using n_domain and prop_exact (min 3)
            self.n_exact = max(int(self.n_domain*self.prop_exact),3)
            # build exact dataset from domain dataset
            np.random.shuffle(index)
            index_exac = index[:self.n_exact]
            self.inputs_exact = inputs[index_exac,:]
            self.U_exact = u[index_exac,:]
            self.F_exact = f[index_exac,:]

        else:
            raise Exception("Only case 1D, 2D, 3D allowed")


    def build_dataset(self):
        """
        Build dataset:
        call the functions to build the dataset
        """
        if(self._flag_dataset_build == False):  # we build the dataset only the first time
            self._load_dataset()
            self._flag_dataset_build = True

    def build_noisy_dataset(self):
        """
        Add noise to exact data
        """
        self.build_dataset()
        if(self._flag_dataset_noise == False):  # we add the noise only the first time
            self.U_with_noise = np.zeros_like(self.U_exact)
            self.F_with_noise = np.zeros_like(self.F_exact)

            for i in range(0,len(self.U_exact)):
                u_error = np.random.normal(0, self.noise_lv, 1)
                self.U_with_noise[i,:] = self.U_exact[i,:] + u_error

                # not so useful... we use onlt U_with_noise, not F (left it for future different implmentations)
                f_error = np.random.normal(0, self.noise_lv, self.F_exact.shape[1]) #1, 2 or 3
                self.F_with_noise[i,:] = self.F_exact[i,:] + f_error

            self._flag_dataset_noise = True

# %% Getters

    def get_dom_data(self):
        """ return domain data"""
        self.build_dataset()
        return self.inputs_dom, self.U_dom, self.F_dom

    def get_coll_data(self):
        """Return collocation data"""
        self.build_dataset()
        return self.inputs_coll,self.U_coll,self.F_coll

    def get_exact_data(self):
        """ return exact data"""
        self.build_dataset()
        return self.inputs_exact, self.U_exact, self.F_exact

    def get_exact_data_with_noise(self):
        """ return exact data + noise """
        self.build_dataset()
        self.build_noisy_dataset()
        return self.inputs_exact, self.U_with_noise, self.F_with_noise

    def get_num_collocation(self):
        """get number of collocation points"""
        self.build_dataset()
        return self.n_collocation

    def get_num_exact(self):
        """get number of exact points"""
        self.build_dataset()
        return self.n_exact

    def get_n_input(self):
        """get n input"""
        return self.n_input
