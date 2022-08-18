import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm
from networks.CoreNN import CoreNN

class Test_Alg(Algorithm):
    """
    Class for Test training
    """
    def __init__(self, bayes_nn, param_method):
        super().__init__(bayes_nn, param_method)

    def sample_theta(self, num):
        self.model.initialize_NN(num+self.model.seed)
        theta = self.model.nn_params
        return theta
