import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm
from networks.CoreNN import CoreNN

class Test_Alg(Algorithm):
    """
    Class for HMC training
    """
    def __init__(self, bayes_nn, dataset):
        super(Test_Alg, self).__init__(bayes_nn, dataset)

    def sample(self):
        par = self.model.par
        theta_1 = CoreNN(par).nn_params
        #print(theta_1)
