import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm
from networks.CoreNN import CoreNN

class Test_Alg(Algorithm):
    """
    Class for HMC training
    """
    def __init__(self, bayes_nn):
        super(Test_Alg, self).__init__(bayes_nn)

    def sample_theta(self, num):
        self.model.initialize_NN(num)
        theta = self.model.nn_params
        print(theta[0][0])
        return theta
