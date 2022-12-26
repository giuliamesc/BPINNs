import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm
from networks import BayesNN  


class SVGD(Algorithm):
    
    def __init__(self): 
        self.__build_particles()

    def __build_particles(self): 
        self.particles = list()

    def sample_theta(self):
        return None

    def select_thetas(self, thetas_train):
        return thetas_train[-1]


class Particle(BayesNN):

    def __init__(self): pass