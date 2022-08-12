import numpy as np
import tensorflow as tf

from Algorithm import Algorithm

class HMC(Algorithm):
    """
    Class for HMC training
    """
    def __init__(self, bayes_nn, dataset):
        
        self.data_train = dataset
        self.model = bayes_nn


    def train(self, par):
        n_epochs = par
        for _ in range(n_epochs):
            new_theta = 0.
            self.bayes_nn.theta.append(new_theta)