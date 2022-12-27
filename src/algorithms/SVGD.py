import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm
from networks import BayesNN  


class SVGD(Algorithm):
    
    def __init__(self, bayes_nn, param_method, debug_flag): 
        super().__init__(bayes_nn, param_method, debug_flag)
        self.burn_in = param_method["burn_in"]
        self.N = param_method["N"]
        self.h = param_method["h"]
        self.__build_particles(param_method["eps"])

    def __build_particles(self, eps): 
        par, equation = self.model.constructors
        seed = self.model.seed + 1
        self.particles = [Particle(par, equation, eps).initialize_NN(seed+n) for n in range(self.N)]

    def __gather_thetas(self):
        return np.array([p.nn_params  for p in self.particles])

    def __gather_grads(self):
        full_loss = self.curr_ep > self.burn_in
        comp_grad = lambda p: p.grad_loss(self.data_batch, full_loss)
        grads  = np.array([comp_grad(p) for p in self.particles])
        return grads

    def __kernel(self):
        rbf = np.vectorize(lambda x: np.exp(-x.ssum()/self.h))
        theta_i, theta_j = np.meshgrid(self.thetas, self.thetas)
        theta_diff = theta_i - theta_j
        K  = rbf(theta_diff)
        GK = np.matmul(K, theta_diff*2/self.h)
        return K, GK

    def __scatter(self, phi):
        for n in range(self.N):
            self.particles[n].update_theta(phi[n])
        return self.__gather_thetas()

    def sample_theta(self):
        self.thetas  = self.__gather_thetas()
        self.grads   = self.__gather_grads()
        K, GK = self.__kernel()
        driver = np.matmul(K, self.grads)
        repuls = np.matmul(GK, np.ones([self.N]))
        phi = (driver + repuls)/self.N
        self.thetas = self.__scatter(phi)
        return self.thetas

    def select_thetas(self, thetas_train):
        return list(thetas_train[-1])


class Particle(BayesNN):

    def __init__(self, par, equation, eps):
        super().__init__(par, equation)
        self.eps = eps

    def update_theta(self, phi):
        self.nn_params = self.nn_params + phi*self.eps