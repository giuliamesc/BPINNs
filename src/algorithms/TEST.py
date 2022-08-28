import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm

class TEST(Algorithm):
    """
    Class for Test training
    """
    def __init__(self, bayes_nn, param_method, debug_flag):
        super().__init__(bayes_nn, param_method, debug_flag)

    def sample_theta(self, num):
        self.model.initialize_NN(num+self.model.seed)
        theta = self.model.nn_params
        return theta

    def select_thetas(self, thetas_train):
        """ Compute burn-in and skip samples """
        return thetas_train