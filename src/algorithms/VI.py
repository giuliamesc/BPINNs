import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm

class VI(Algorithm):
    """
    Class for VI training
    """
    def __init__(self, bayes_nn, param_method):
        super().__init__(bayes_nn, param_method)