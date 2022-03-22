import numpy as np
import matplotlib.pyplot as plt
import os

class AnalyticalData:
    def __init__(self, test_case, do_plots = False, test_only = False, save_plot = False):
        self.test_case = test_case
        self.do_plots  = do_plots
        self.test_only = test_only
        self.save_plot = save_plot

        self.solution = analytical_solution[self.test_case]
        self.domain_data = analytical_domain[self.test_case]
        self.dimension = len(self.domain_data["domain"])
        self.save_folder = 'data'

        self._creating_loop()

    def _creating_loop(self):
        print(f"\n\nGenerating dataset: {self.test_case}")
        self._build_directory()
        self._create_domain()
        self._create_solutions()
        if self.do_plots:
            self._plotter()
        print(f"Dataset {self.test_case} generated!")

    def _create_solutions(self):
        for key,_ in self.solution.items():
            self._create_sol(key)

    def _create_sol(self, name):
        func = self.solution[name]
        grid_list = np.split(self.grid, self.dimension, axis = 0)
        grid = [x.squeeze() for x in grid_list]
        sol = func(*grid)
        self._save_data(name, sol)

    def _create_domain(self):
        if self.domain_data["mesh_type"] == "uniform":
            self._create_uniform_domain()
        elif domain_data["mesh_type"] == "sobol":
            self._create_sobol_domain()
        else:
            raise Exception("This mesh type doesn't exists")


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

    def _create_sobol_domain(self):
        pass

    def _save_data(self, name, data):
        filename = os.path.join(self.save_path,name)
        np.save(filename,data)
        print(f'Saved {name}')

    def _build_directory(self):
        if self.test_only:
            self.save_folder = os.path.join(self.save_folder,"trash")
        self.save_path = os.path.join(self.save_folder,self.test_case)
        if not(os.path.isdir(self.save_path)):
            os.mkdir(self.save_path)
            print(f'Folder {self.save_path} created')
        else:
            print('Folder already present')

    def _plot(self, var_name):
        load = lambda name : np.load(os.path.join(self.save_path, f'{name}.npy'))
        var = load(var_name)
        plt.figure()
        if self.dimension == 1:
            var_dim = "(x)"
            plt.plot(load('x'),var,'b')
        elif self.dimension == 2:
            var_dim = "(x,y)"
            plt.scatter(load('x'), load('y'), c = var, cmap = 'coolwarm', 
                        vmin = min(var), vmax = max(var))
            plt.colorbar()
        else:
            print("Plotter non available")
        plt.title(f'{var_name}{var_dim}')
        if self.save_plot:
            plt.savefig(os.path.join(self.save_path,f"{var_name}.png"))

    def _plotter(self):
        for var_file in os.listdir(self.save_path):
            if var_file[-4:] == ".npy":
                var_name = var_file[:-4]
                if var_name not in ["x","y","z"]:
                    self._plot(var_name)


analytical_domain = {
    "test_ell_cos1d": {
        "mesh_type": "uniform",
        "resolution": 100,
        "domain": [(0,8)]
        },
    "test_ell_cos2d": {
        "mesh_type": "uniform",
        "resolution": 100,
        "domain": [(0,8),(0,6)]
        }
    }

analytical_solution = {
    "test_ell_cos1d": {
        "u": lambda *x: np.cos(x[0]),
        "f": lambda *x: np.cos(x[0])
        },
    "test_ell_cos2d": {
        "u": lambda *x: np.cos(x[0]) + np.cos(x[1]),
        "f": lambda *x: np.cos(x[0]*x[1])
        }
    }

if __name__ == "__main__":
    AnalyticalData("test_ell_cos1d", do_plots = True, test_only = True, save_plot = True)
    AnalyticalData("test_ell_cos2d", do_plots = True, test_only = True, save_plot = True)
