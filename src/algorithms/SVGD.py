import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm

class SVGD(Algorithm):
    """
    Class for SVGD training
    """
    def __init__(self, bayes_nn, param_method):
        super().__init__(bayes_nn, param_method)

    def sample_theta(self):
        raise Exception("Work in Progress")
        return None