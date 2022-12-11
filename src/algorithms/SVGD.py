import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm
from networks import BayesNN  

class SVGD(Algorithm):
    """
    Class for SVGD training
    """
    def __init__(self, bayes_nn, param_method, debug_flag):
        super().__init__(bayes_nn, param_method, debug_flag)
        self.N = param_method["N"]
        self.particles = [self.__build_particle(bayes_nn, i+1) for i in range(self.N)]
        import pdb; pdb.set_trace()
        # Compute Loss e Grad Loss
        # Compute kernels k
        # Compute update func
        # Update theta

    def __build_particle(self, bayes_nn, seed):
        seed += bayes_nn.seed
        par, equation = bayes_nn.constructors
        particle = BayesNN(par, equation)
        particle.initialize_NN(seed+bayes_nn.seed)
        return particle

    def __gather_thetas(self):
        pass

    def __gather_grads(self): 
        pass

    def __kernel(self):
        pass

    def __phi(self):
        pass  

    def sample_theta(self):
        old_theta  = self.__gather_thetas()
        grads = self.__gather_grads()
        return None

    def select_thetas(self, thetas_train):
        """ Compute burn-in and skip samples """
        #for p in self.particles: del p
        return thetas_train[-1]