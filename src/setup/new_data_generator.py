import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

class AnalyticalData:
    def __init__(self, data_config):
        
        self.test_case = data_config.name
        self.problem   = data_config.problem
        self.save_folder = '../data'

        self.mesh      = data_config.mesh
        self.domains   = data_config.domains
        self.values    = data_config.values
        self.dimension = len(self.domains["full"])

        self.create_domains()

        ### TEST ZONE
        self.plot(self.__create_multidomain(self.domains["sol"], self.mesh["inner_res"]))

    def compute_bnd(self, bnd):
        l_bounds = [i[0] for i in bnd]
        u_bounds = [i[1] for i in bnd]
        return (l_bounds, u_bounds)

    def create_domains(self):
        self.create_dom_pde()
        self.create_dom_sol()
        self.create_dom_par()
        self.create_dom_bnd()
        
    def create_dom_sol(self): # u_train
        # multiX, U(X), Save
        pass
    def create_dom_par(self): # f_train
        # multiX, F(X), Save
        pass
    def create_dom_pde(self): # pde
        # singleX, Save 
        pass
    def create_dom_bnd(self): # bnd
        # ?X, project, merge edges, U(X), Save
        pass
    def create_test(self): # u_test, f_test
        # singleX+uniform, U(X), F(X), Save U, F
        pass

    def __merge_2points(self, p1, p2, n1, n2): 
        if n1 < n2: p1, p2, n1, n2 = p2, p1, n2, n1
        i, j = 0, 0
        r = n1 / n2
        points = np.zeros([self.dimension, n1+n2])
        for k in range(n1+n2):
            choice = i/(j+1e-8) < r
            points[:,k], i, j = (p1[:,i], i+1, j) if choice else (p2[:,j], i, j+1)
        return points, n1+n2

    def __create_multidomain(self, bnd, num):
        dim_dom   = [np.prod([d[1]-d[0] for d in d_bnd]) for d_bnd in bnd] 
        num_dom   = [(dd*num)//sum(dim_dom) for dd in dim_dom]
        multi_pts = [self.__create_domain(d_bnd, d_num) for d_bnd, d_num in zip(bnd, num_dom)]
        pts, num = multi_pts[0], num_dom[0]
        for p,n in zip(multi_pts[1:], num_dom[1:]):
            pts, num = self.__merge_2points(p, pts, n, num)
        return pts

    def __create_domain(self, bnd, num, mesh=None):
        if mesh is None: mesh = self.mesh["mesh_type"]
        bnd = self.compute_bnd(bnd)
        match mesh:
            case "uniform": return self.__create_uniform_domain(bnd, num)
            case "random" : return self.__create_random_domain(bnd, num)
            case "sobol"  : return self.__create_sobol_domain(bnd, num)
            case _ : Exception("This mesh type doesn't exists")

    ### DA RISCRIVERE PER DATASET DI TEST
    def __create_uniform_domain(self, bnd, num):
        x_pts = list()
        for i in range(self.dimension): 
            a, b, n = bnd[i][0], bnd[i][1], num[i]
            x_pts.append(np.linspace(a, b, n+1)[:-1] + (b-a)/(2*n))
        if self.dimension == 2: x = np.meshgrid(x_pts[0],x_pts[1])
        if self.dimension == 3: x = np.meshgrid(x_pts[0],x_pts[1],x_pts[2])
        points = np.zeros([self.dimension, np.prod(num)])
        for i, v in enumerate(x): points[i,:] = np.reshape(v,[np.prod(num)])
        return points

    def __create_random_domain(self, bnd, num):
        x_square = np.random.rand(num, self.dimension)
        return qmc.scale(x_square, bnd[0], bnd[1]).T

    def __create_sobol_domain(self, bnd, num): 
        sobolexp = int(np.ceil(np.log(num)/np.log(2)))
        sampler = qmc.Sobol(d=self.dimension, scramble=False)
        sample = sampler.random_base2(m=sobolexp)
        sample += (1./float(2**sobolexp))/2.
        return qmc.scale(sample, bnd[0], bnd[1]).T

    def plot(self, points):
        if self.dimension == 1: plotter = self.plot1D
        if self.dimension == 2: plotter = self.plot2D
        plotter(self.compute_bnd(self.domains["full"]), points)

    def plot1D(self, bnd, points):
        plt.figure()
        plt.xlim([bnd[0][0], bnd[1][0]])
        plt.plot(points[0,:], [0]*points[0,:], "*")
        plt.show()

    def plot2D(self, bnd, points):
        plt.figure()
        plt.xlim([bnd[0][0], bnd[1][0]]) 
        plt.ylim([bnd[0][1], bnd[1][1]])
        plt.plot(points[0,:], points[1,:], "*")
        plt.show()

    def __save_data(self, name, data):
        pass