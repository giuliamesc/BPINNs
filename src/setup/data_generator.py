from utility import create_data_folders, starred_print
import matplotlib.pyplot as plt
from scipy.stats import qmc
import numpy as np
import warnings
import os

class DataGenerator:
    def __init__(self, data_config, main=True):
        
        self.test_case = data_config.name
        self.problem   = data_config.problem
        self.save_folder = '../data'
        self.main = main

        self.mesh    = data_config.mesh
        self.domains = data_config.domains
        self.values  = data_config.values
        self.dim     = len(self.domains["full"])

        print(f"\tGenerating dataset: {self.test_case}")
        self.save_path = create_data_folders(self.problem , self.test_case, self.main)
        self.__create_domains() # Main Creation Loop
        print(f"\tDataset {self.test_case} generated")
        
        if not self.main: return plt.show()

    def __compute_bnd(self, bnd):
        " Separate lower and upper bounds of a domain "
        l_bounds = [i[0] for i in bnd]
        u_bounds = [i[1] for i in bnd]
        return (l_bounds, u_bounds)

    def __create_domains(self):
        " call all the creating functions "
        self.__create_dom_pde()
        self.__create_dom_sol()
        self.__create_dom_par()
        self.__create_dom_bnd()
        self.__create_test()
        
    def __create_dom_sol(self):
        " Create and save: dom_sol, sol_train "
        X = self.__create_multidomain(self.domains["sol"], self.mesh["inner_res"])
        self.__save_data("dom_sol", X)
        self.__save_data("sol_train", np.array(self.values["u"](X.T)).T)
        if not self.main: self.plot(X, c="g")

    def __create_dom_par(self):
        " Create and save: dom_par, par_train "
        X = self.__create_multidomain(self.domains["par"], self.mesh["inner_res"])
        self.__save_data("dom_par", X)
        self.__save_data("par_train", np.array(self.values["f"](X.T)).T)
        if not self.main: self.plot(X, c="b")

    def __create_dom_pde(self):
        " Create and save: dom_pde "
        X = self.__create_domain(self.domains["full"], self.mesh["inner_res"])
        self.__save_data("dom_pde",X)

    def __create_dom_bnd(self):
        " Create and save: dom_bnd, sol_bnd "
        lu_bnd, d = self.__compute_bnd(self.domains["full"]), self.dim
        points = self.__create_domain(self.domains["full"], self.mesh["outer_res"], "sobol")
        X_list = [points.copy() for _ in range(2*d)]
        for i in range(d):
            X_list[i  ][i,:] = lu_bnd[0][i]
            X_list[i+d][i,:] = lu_bnd[1][i]
        X = np.zeros([d, self.mesh["outer_res"]*2*d])
        for i in range(2*d): X[:,i+0::2*d] = X_list[i+0]
        self.__save_data("dom_bnd",X)
        self.__save_data("sol_bnd", np.array(self.values["u"](X.T)).T)
        if not self.main: self.plot(X, c="r")

    def __create_test(self):
        " Create and save: dom_test, sol_test, par_test"
        X = self.__create_domain(self.domains["full"], self.mesh["test_res"], "uniform")
        self.__save_data("dom_test", X)
        self.__save_data("sol_test", np.array(self.values["u"](X.T)).T)
        self.__save_data("par_test", np.array(self.values["f"](X.T)).T)

    def __merge_2points(self, p1, p2, n1, n2): 
        " Merge two sequences of points altrernating the points"
        if n1 < n2: p1, p2, n1, n2 = p2, p1, n2, n1
        i, j, r = 0, 0, n1/n2
        points = np.zeros([n1+n2, self.dim])
        for k in range(n1+n2):
            choice = i/(j+1e-8) < r
            points[k,:], i, j = (p1[i,:], i+1, j) if choice else (p2[j,:], i, j+1)
        return points, n1+n2

    def __create_multidomain(self, bnd, num, mesh=None):
        " Split multi-domain bnd and call single domain creator "
        if mesh is None: mesh = self.mesh["mesh_type"]
        dim_dom   = [np.prod([d[1]-d[0] for d in d_bnd]) for d_bnd in bnd] 
        num_dom   = [int((dd*num)//sum(dim_dom)) for dd in dim_dom]
        multi_pts = [self.__create_domain(d_bnd, d_num, mesh) for d_bnd, d_num in zip(bnd, num_dom)]
        pts, num = multi_pts[0], num_dom[0]
        for p,n in zip(multi_pts[1:], num_dom[1:]): pts, num = self.__merge_2points(p, pts, n, num)
        return pts

    def __create_domain(self, bnd, num, mesh=None):
        " Caller of specific mesh creator in bnd domain with num elements "
        if mesh is None: mesh = self.mesh["mesh_type"]
        bnd = self.__compute_bnd(bnd)
        match mesh:
            case "uniform": return self.__create_uniform_domain(bnd, num)
            case "random" : return self.__create_random_domain(bnd, num)
            case "sobol"  : return self.__create_sobol_domain(bnd, num)
            case _ : Exception("This mesh type doesn't exists")

    def __create_uniform_domain(self, bnd, num):
        " Create a uniform mesh in bnd domain with num elements "
        x_line = np.linspace(0,1,num+1)[:-1] + 1/(2*num)
        if self.dim == 1: x = np.meshgrid(x_line)
        if self.dim == 2: x = np.meshgrid(x_line,x_line)
        if self.dim == 3: x = np.meshgrid(x_line,x_line,x_line)
        x_square = np.zeros([num**self.dim, self.dim])
        for i, v in enumerate(x): x_square[:,i] = np.reshape(v,[num**self.dim])
        return qmc.scale(x_square, bnd[0], bnd[1])

    def __create_random_domain(self, bnd, num):
        " Create a random mesh in bnd domain with num elements "
        x_square = np.random.rand(num, self.dim)
        return qmc.scale(x_square, bnd[0], bnd[1])

    def __create_sobol_domain(self, bnd, num):
        " Create a sobol mesh in bnd domain with num elements "
        if ((num) & (num-1)): warnings.warn("Non optimal choice of resolution for Sobol mesh")
        sobolexp = int(np.ceil(np.log(num)/np.log(2)))
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        sample = sampler.random_base2(m=sobolexp)
        sample += (1./float(2**sobolexp))/2.
        return qmc.scale(sample, bnd[0], bnd[1])

    def __save_data(self, name, data):
        """ Saves the data generated """
        filename = os.path.join(self.save_path, name)
        np.save(filename, data)

    ### TEMP FILES

    def plot(self, points, c="b"):
        if self.dim == 1: plotter = self.plot1D
        if self.dim == 2: plotter = self.plot2D
        plotter(self.__compute_bnd(self.domains["full"]), points, c)

    def plot1D(self, bnd, points, c="b"):
        plt.xlim([bnd[0][0], bnd[1][0]])
        plt.plot(points[0,:], [0]*points[0,:], f"{c}*")

    def plot2D(self, bnd, points, c="b"):
        plt.xlim([bnd[0][0], bnd[1][0]]) 
        plt.ylim([bnd[0][1], bnd[1][1]])
        plt.plot(points[0,:], points[1,:], f"{c}*")