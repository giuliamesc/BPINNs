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
    """

    def __init__(self, par):
        
        self.pde_type = par.pde
        self.problem  = par.problem
        self.name_example = par.folder_name
        self.mesh_type = par.config.analytical_domain["mesh_type"]

        self.num_fitting     = par.experiment["num_fitting"]
        self.num_collocation = par.experiment["num_collocation"]
        self.num_boundary    = 1 # To be generalized in 2D case
        self.noise_lv = par.experiment["var_data"]
        
        self.n_input   = par.phys_dim.n_input
        self.n_out_par = par.phys_dim.n_out_par
        self.n_out_sol = par.phys_dim.n_out_sol

        np.random.seed(par.utils["random_seed"])
        self.dom_data = self.__load_dataset()
        self.__compute_norm_coeff()


    @property
    def dom_data(self):
        return self.x_dom, self.u_dom, self.f_dom
    
    @dom_data.setter
    def dom_data(self, values):
        self.x_dom = values[0]
        self.u_dom = values[1]
        self.f_dom = values[2]

    @property
    def coll_data(self):
        return self.x_coll, self.u_coll, self.f_coll

    @coll_data.setter
    def coll_data(self, indexes):
        dom_val = self.dom_data
        self.x_coll = dom_val[0][indexes,:]
        self.u_coll = dom_val[1][indexes,:]
        self.f_coll = dom_val[2][indexes,:]

    @property
    def exact_data(self):
        return self.x_exact, self.u_exact, self.f_exact
    
    @exact_data.setter
    def exact_data(self, indexes):
        dom_val = self.dom_data
        self.x_exact = dom_val[0][indexes,:]
        self.u_exact = dom_val[1][indexes,:]
        self.f_exact = dom_val[2][indexes,:]

    @property
    def noise_data(self):
        return self.x_exact, self.u_noise, self.f_noise

    @noise_data.setter
    def noise_data(self, noise):
        data = self.exact_data
        self.u_noise = np.random.normal(data[1], noise, data[1].shape).astype("float32")
        self.f_noise = np.random.normal(data[2], noise, data[2].shape).astype("float32")

    @property
    def norm_coeff(self):
        u_coeff = (self.mean_u, self.std_u)
        f_coeff = (self.mean_f, self.std_f)
        return u_coeff, f_coeff

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

    def __load_dataset(self):

        path = os.path.join("../data", self.problem)
        path = os.path.join(path, self.name_example)

        x = list()
        for var_file in os.listdir(path):
            if not var_file[-4:] == ".npy": continue
            if var_file[:-4] in ["x","y","z"]:
                x.append(np.load(os.path.join(path,var_file))[...,None])
        x = np.concatenate(x, axis=-1)
        x = x.astype(np.float32)

        u = np.load(os.path.join(path,"u.npy")).astype(np.float32)
        f = np.load(os.path.join(path,"f.npy")).astype(np.float32)

        if(len(u.shape)==1): u = u[...,None] # from shape (n_coll, ) -> (n_coll, 1)
        if(len(f.shape)==1): f = f[...,None] # add the last dimension only if we are in 1D case

        return x, u, f

    def __compute_norm_coeff(self):
        data = self.dom_data
        self.mean_u, self.std_u = np.mean(data[1],axis=0), np.std(data[1],axis=0)
        self.mean_f, self.std_f = np.mean(data[2],axis=0), np.std(data[2],axis=0)

    def __build_dataset(self):
        num_points = self.dom_data[0].shape[0]
        if self.num_collocation > num_points : 
            raise Exception(f'Num collocation cannot be bigger than dataset size: {self.num_collocation} > {num_points}')
        if self.num_fitting     > num_points : 
            raise Exception(f'Num fitting cannot be bigger than dataset size: {self.num_fitting} > {num_points}')
        # build collocation dataset from domain dataset
        self.coll_data  = self.__select_indexes(num_points, self.num_collocation)
        # build exact dataset from domain dataset 
        self.exact_data = self.__select_indexes(num_points, self.num_fitting)
        # build noise dataset from domain dataset 
        self.noise_data = self.noise_lv

    def normalize_dataset(self):
        data = self.dom_data
        import pdb; pdb.set_trace()
        u_star = (data[1]-self.norm_coeff[0][0])/self.norm_coeff[0][1]
        f_star = (data[2]-self.norm_coeff[1][0])/self.norm_coeff[1][1]
        self.dom_data = data[0], u_star, f_star
        self.__build_dataset()

    def denormalize_dataset(self):
        data = self.dom_data
        u = data[1] * self.norm_coeff[0][1] + self.norm_coeff[0][0]
        f = data[2] * self.norm_coeff[1][1] + self.norm_coeff[1][0]
        self.dom_data = data[0], u, f
        self.__build_dataset()