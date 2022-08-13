import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm

class HMC(Algorithm):
    """
    Class for HMC training
    """
    def __init__(self, bayes_nn):
        super().__init__(bayes_nn)

    def sample_theta(self):
        print("Sampling")
