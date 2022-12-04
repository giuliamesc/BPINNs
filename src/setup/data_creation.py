import os
import numpy as np

class Dataset:
    def __init__(self, par):
        self.pde_type = par.pde
        self.problem  = par.problem
        self.name_example = par.folder_name

        self.num_points  = par.num_points
        self.uncertainty = par.uncertainty

        np.random.seed(par.utils["random_seed"])

        self.__load_dataset() 
        self.__compute_norm_coeff()
        self.__add_noise()

    def __load_dataset(self):
        self.path = os.path.join("../data", self.problem)
        self.path = os.path.join(self.path, self.name_example)
        load = lambda name : np.load(os.path.join(self.path,name))
        self.__data_all = {name[:-4]: load(name) for name in os.listdir(self.path)}

    @property
    def data_all(self):
        return self.__data_all

    @property
    def data_sol(self):
        selected, num = ("dom_sol","sol_train"), self.num_points["sol"]
        return {k[:3]: self.data_all[k][:,:num] for k in selected}

    @property
    def data_par(self):
        selected, num = ("dom_par","par_train"), self.num_points["par"]
        return {k[:3]: self.data_all[k][:,:num] for k in selected}

    @property
    def data_bnd(self):
        selected, num = ("dom_bnd","sol_bnd"), self.num_points["bnd"]
        return {k[:3]: self.data_all[k][:,:num] for k in selected}

    @property
    def data_pde(self):
        selected, num = ("dom_pde"), self.num_points["pde"]
        return {k[:3]: self.data_all[k][:,:num] for k in selected}

    @property
    def data_test(self):
        selected = ("dom_test","sol_test","par_test")
        return {k[:3]: self.data_all[k] for k in selected}

    @data_all.setter
    def data_all(self, items):
        name, values = items
        self.__data_all[name] = values

    @data_sol.setter
    def data_sol(self, items): self.data_all = items
    
    @data_par.setter
    def data_par(self, items): self.data_all = items
    
    @data_bnd.setter
    def data_bnd(self, items): self.data_all = items
    
    @data_pde.setter # Unused
    def data_pde(self, items): self.data_all = items
    
    @data_test.setter # Unused
    def data_test(self, items): self.data_all = items

    @property
    def data_plot(self):
        plots = dict()
        plots["sol_ex"] = (self.data_test["dom"], self.data_test["sol"])
        plots["sol_ns"] = ( self.data_sol["dom"],  self.data_sol["sol"])
        plots["par_ex"] = (self.data_test["dom"], self.data_test["par"])
        plots["par_ns"] = ( self.data_par["dom"],  self.data_par["par"])
        plots["bnd_ns"] = ( self.data_bnd["dom"],  self.data_bnd["sol"])
        return plots

    def normalize_dataset(self):
        for key in self.data_all:
            if key.startswith("dom"): continue
            mean, std = self.norm_coeff[f"{key[:3]}_mean"], self.norm_coeff[f"{key[:3]}_std"]
            self.__data_all[key] = (self.__data_all[key] - mean) / std

    def denormalize_dataset(self):
        for key in self.data_all:
            if key.startswith("dom"): continue
            mean, std = self.norm_coeff[f"{key[:3]}_mean"], self.norm_coeff[f"{key[:3]}_std"]
            self.__data_all[key] = self.__data_all[key] * std + mean

    def __compute_norm_coeff(self):
        self.norm_coeff = dict()
        self.norm_coeff["sol_mean"] = np.mean(self.data_test["sol"], axis=0)
        self.norm_coeff["sol_std" ] =  np.std(self.data_test["sol"], axis=0)
        self.norm_coeff["par_mean"] = np.mean(self.data_test["par"], axis=0)
        self.norm_coeff["par_std" ] =  np.std(self.data_test["par"], axis=0)

    def __add_noise(self):
        noise_values = lambda x,y: np.random.normal(x, y, x.shape).astype("float32") 
        self.data_sol = ("sol_train", noise_values(self.data_sol["sol"], self.uncertainty["sol"]))
        self.data_par = ("par_train", noise_values(self.data_par["par"], self.uncertainty["par"]))
        self.data_bnd = ("sol_bnd"  , noise_values(self.data_bnd["sol"], self.uncertainty["bnd"]))