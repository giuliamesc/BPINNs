import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm

class HMC(Algorithm):
    """
    Class for HMC training
    """
    def __init__(self, bayes_nn, dataset):
        super().__init__(bayes_nn, dataset)

    def sample_theta(self):
        print("Sampling")
