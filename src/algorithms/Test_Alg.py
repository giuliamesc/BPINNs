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

    def sample_theta(self):
        par = self.model.par
        theta = CoreNN(par).nn_params
        return theta
