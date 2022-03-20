import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import os

class AnalyticalData:
    def __init__(self, test_case, do_plots = False):
        self.test_case = test_case
        self.do_plots = do_plots
        self.solution = analytical_solution[self.test_case]
        self.domain_data = analytical_domain[self.test_case]
        self.dimension = len(self.domain_data["domain"])
        self.save_folder = '../data/'
    
        
        self._creating_loop()
        
    def _creating_loop(self):
        self._build_directory()
        self._create_domain()
        self._create_solutions()
        if self.do_plots:
            self._plotter()
        
    def _create_solutions(self):
        for key,_ in self.solution.items():
            self._create_sol(key)

        
    def _create_domain(self):
        self._create_uniform_domain()

        
    def _create_uniform_domain(self):
        resolution = self.domain_data["resolution"]
        x = np.zeros([self.domain_data["resolution"], self.dimension])
        
        for i in range(self.dimension):
            x[:,i] = np.linspace(self.domain_data["domain"][i][0], self.domain_data["domain"][i][1], resolution)
        if self.dimension == 2:
            x = np.meshgrid(x[:,0],x[:,1])
        if self.dimension == 3:
            x = np.meshgrid(x[:,0],x[:,1],x[:,2])
                
        self.grid = np.reshape(x,[self.dimension, resolution**self.dimension])
        names = ["x","y","z"]
        for i in range(self.dimension):
            self._save_data(names[i], self.grid[i,:])
    
    def _create_sol(self, name):
        func = self.solution[name]
        grid_list = np.split(self.grid, self.dimension, axis = 0)
        grid = [x.squeeze() for x in grid_list]
        if self.dimension == 1:
            sol = func(grid[0])
        if self.dimension == 2:
            sol = func(grid[0],grid[1])
        
        self._save_data(name, sol)
    
    def _save_data(self, name, data):
        filename = os.path.join(self.save_path,name)
        np.save(filename,data)
        print(f'Saved {name}')
    
    def _build_directory(self):
        self.save_path = os.path.join(self.save_folder,self.test_case)
        if not(os.path.isdir(self.save_path)):
            os.mkdir(self.save_path)
            print(f'Folder {self.save_path} created')
        else:
            print('Folder already present')
    
    def _plotter(self):
        load = lambda name : np.load(os.path.join(self.save_path, f'{name}.npy'))
        if self.dimension == 1:
            plt.plot(load('x'),load('u'),'m')
            plt.title('u(x)')
            plt.plot(load('x'),load('f'),'b')
            plt.title('f(x)')
        if self.dimension == 2:
            plt.figure()
            plt.scatter(load('x'), load('y'), c= load('u'), label = 'u', 
                        cmap = 'coolwarm', vmin = min(load('u')), vmax = max(load('u')))
            plt.title('u(x,y)')
            plt.colorbar()
            plt.figure()
            plt.scatter(load('x'), load('y'), c= load('f'), label = 'f', 
                        cmap = 'coolwarm', vmin = min(load('f')), vmax = max(load('f')))
            plt.title('f(x,y)')
            plt.colorbar()
        
        
analytical_domain = {
    "ell_cos1d": {
        "resolution": 100,
        "domain": [(0,8)]
        },
    "ell_cos2d": {
        "resolution": 100,
        "domain": [(0,8),(0,6)]
        }
    }

analytical_solution = {
    "ell_cos1d": {
        "u": lambda x: np.cos(x),
        "f": lambda x: np.cos(x)
        },
    "ell_cos2d": {
        "u": lambda x,y: np.cos(x) + np.cos(y),
        "f": lambda x,y: np.cos(x*y)
        }
    }
    
if __name__ == "__main__":
    AnalyticalData("ell_cos2d", True)
