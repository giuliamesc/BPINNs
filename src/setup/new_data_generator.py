import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

class AnalyticalData:
    def __init__(self, data_config):
        
        self.test_case = data_config.name
        self.problem   = data_config.problem
        self.save_folder = '../data'

        self.domains = data_config.domains
        self.dimension = len(self.domains["full"])
        num = self.domains["inner_res"]
        bnd = self.domains["full"]

        #self.solution = data_config.analytical_solution
    
        #self.create_domains()
        #self.__create_uniform_domain([(0,2),(0,3)],[4,6])
        self.__create_sobol_domain(bnd,num)

    def create_domains(self):
        self.create_dom_pde()
        self.create_dom_sol()
        self.create_dom_par()
        self.create_dom_bnd()
        
    def create_dom_pde():
        pass
    def create_dom_sol():
        pass
    def create_dom_par():
        pass
    def create_dom_bnd():
        pass

    def __create_multidomain(self, bnd, num):
        pass

    def __create_rectangle(self, bnd, num):
        match self.domains["mesh_type"]:
            case "uniform": return self.__create_uniform_domain(bnd, num)
            case "random" : return self.__create_random_domain(bnd, num)
            case "sobol"  : return self.__create_sobol_domain(bnd, num)
            case _ : Exception("This mesh type doesn't exists")

    def __create_uniform_domain(self, bnd, num):
        x_pts = list()
        for i in range(self.dimension): 
            a, b, n = bnd[i][0], bnd[i][1], num[i]
            x_pts.append(np.linspace(a, b, n+1)[:-1] + (b-a)/(2*n))
        if self.dimension == 2: x = np.meshgrid(x_pts[0],x_pts[1])
        if self.dimension == 3: x = np.meshgrid(x_pts[0],x_pts[1],x_pts[2])
        points = np.zeros([self.dimension, np.prod(num)])
        for i, v in enumerate(x): points[i,:] = np.reshape(v,[np.prod(num)])
        total, part = len(points[0]), 12
        self.plot(bnd, points)
        return points

    def __create_random_domain(self, bnd, num): pass
    def __create_sobol_domain(self, bnd, num): 
        l_bounds = [i[0] for i in bnd]
        u_bounds = [i[1] for i in bnd]
        sobolexp = int(np.ceil(np.log(num)/np.log(2)))
        sampler = qmc.Sobol(d=self.dimension, scramble=False)
        sample = sampler.random_base2(m=sobolexp)
        sample += (1./float(2**sobolexp))/2.
        points = qmc.scale(sample, l_bounds, u_bounds).T
        self.plot(bnd, points)

    def plot(self, bnd, points):
        plt.figure()
        plt.xlim(bnd[0])
        plt.ylim(bnd[1])
        plt.plot(points[0,:], points[1,:], "*")
        plt.show()