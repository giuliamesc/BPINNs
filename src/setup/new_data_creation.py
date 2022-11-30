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
    