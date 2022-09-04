import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc


class AnalyticalData:
    """
    - Generates the dataset from an analytical solution and stores points and functions in .npy files

    Attributes:
        - test_case: Test case used
        - do_plots: Boolean to generate plots
        - test_only: Boolean to consider the current run as test
        - save_plot: Boolean to save plots
        - solution: Solution and parametric field
        - domain_data: Spatial coordinates
        - dimension: Dimension of the input space
        - save_folder: Path to store data
    
    """
    def __init__(self, data_config, gui_len, do_plots=False, test_only=False, save_plot=False, is_main=False):
      
        self.test_case = data_config.name
        self.gui_len   = gui_len

        self.do_plots  = do_plots
        self.test_only = test_only
        self.save_plot = save_plot
        self.is_main   = is_main
        
        self.solution    = data_config.analytical_solution
        self.domain_data = data_config.analytical_domain
        self.dimension = len(self.domain_data["domain"])
        self.save_folder = '../data'

        self.__creating_loop()

    def __creating_loop(self):
        """ Function for dataset generation; creation of folder, domain and solution .npy files, plot generation """
        if self.is_main:
            print(f" Generating dataset: {self.test_case} ".center(self.gui_len,'*'))
        self.__build_directory()
        self.__create_domain()
        self.__create_solutions()
        if self.do_plots:
            self.__plotter()
        print(f"Dataset {self.test_case} generated")

    # %% Build Solution
    def __create_solutions(self):
        """ Creates solution and parametric field files """
        for key,_ in self.solution.items():
            self.__create_sol(key)

    def __create_sol(self, name):
        """ Generates name vector and saves it """
        func = self.solution[name]
        grid_list = np.split(self.grid, self.dimension, axis = 0)
        grid = [x.squeeze() for x in grid_list]
        sol = func(*grid)
        self.__save_data(name, sol)

    # %% Build Domain
    def __create_domain(self):
        """ Creates the spatial domain with the random points generation technique chosen """
        if self.domain_data["mesh_type"] == "uniform":
            self.__create_uniform_domain()
        elif self.domain_data["mesh_type"] == "sobol":
            self.__create_sobol_domain()
        else:
            raise Exception("This mesh type doesn't exists")

    def __create_uniform_domain(self):
        """ Used in __create_domain; generation of a uniform spatial mesh """
        x = np.zeros([self.domain_data["resolution"], self.dimension])

        for i in range(self.dimension):
            x[:,i] = np.linspace(self.domain_data["domain"][i][0], self.domain_data["domain"][i][1],
                                 self.domain_data["resolution"])
        if self.dimension == 2:
            x = np.meshgrid(x[:,0],x[:,1])
        if self.dimension == 3:
            x = np.meshgrid(x[:,0],x[:,1],x[:,2])

        self.grid = np.reshape(x,[self.dimension, self.domain_data["resolution"]**self.dimension])
        names = ["x","y","z"]
        for i in range(self.dimension):
            self.__save_data(names[i], self.grid[i])

    def __create_sobol_domain(self):
        """ Used in __create_domain; generation of a spatial mesh using Sobol points"""
        l_bounds = [i[0] for i in self.domain_data["domain"]]
        u_bounds = [i[1] for i in self.domain_data["domain"]]
        sobolexp = int(np.ceil(np.log(self.domain_data["resolution"])/np.log(2)))
        sampler = qmc.Sobol(d=self.dimension, scramble=False)
        sample = sampler.random_base2(m=sobolexp)
        sample = np.concatenate((sample, np.array([[1.]*self.dimension])))
        self.grid = qmc.scale(sample, l_bounds, u_bounds).T

        names = ["x","y","z"]
        for i in range(self.dimension):
            self.__save_data(names[i], self.grid[i])


    # %% Loading and Saving
    def __save_data(self, name, data):
        """ Saves the data generated """
        filename = os.path.join(self.save_path,name)
        np.save(filename,data)
        if self.is_main:
            print(f'\tSaved {name}')

    def __build_directory(self):
        """ Builds a directory for the case study (if it is a test, the trash folder) """
        if self.test_only:
            trash_path = os.path.join(self.save_folder,"trash")
            if not(os.path.isdir(trash_path)):
                os.mkdir(trash_path)
            self.save_folder = os.path.join(trash_path)
        self.save_path = os.path.join(self.save_folder,self.test_case)
        if self.is_main:
            if not(os.path.isdir(self.save_path)):
                os.mkdir(self.save_path)
                print(f'Folder {self.save_path} created')
            else:
                print('Folder already present')

    def __load(self,name):
        """ Loader of .npy files """
        return np.load(os.path.join(self.save_path, f'{name}.npy'))

    # %% Postprocess
    def __plot(self, var_name):
        """ Used in __plotter; plots var_name profile (distinguishing 1D, 2D, 3D case) """
        var = self.__load(var_name)
        plt.figure()
        if self.dimension == 1:
            var_dim = "(x)"
            plt.scatter(self.__load('x'), var, c = 'b', s = 0.1)
        elif self.dimension == 2:
            var_dim = "(x,y)"
            plt.scatter(self.__load('x'), self.__load('y'), c = var, cmap = 'coolwarm',
                        vmin = min(var), vmax = max(var))
            plt.colorbar()
        else:
            print("Plotter non available")
        plt.title(f'{var_name}{var_dim}')
        if self.save_plot:
            plt.savefig(os.path.join(self.save_path,f"{var_name}.png"))

    def __plotter(self):
        """ Plotter """
        if self.dimension == 2:
            plt.figure()
            plt.scatter(self.__load("x"),self.__load("y"), s=1)
            plt.title("{} mesh".format(self.domain_data["mesh_type"]))
        for var_file in os.listdir(self.save_path):
            if var_file[-4:] == ".npy":
                var_name = var_file[:-4]
                if var_name not in ["x","y","z"]:
                    self.__plot(var_name)

    def show_plot(self):
        """ Shows the plots """
        print(f" END ".center(self.gui_len,'*'))
        plt.show(block = True)