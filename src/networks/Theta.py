import tensorflow as tf
import numpy as np

class Theta():
    """ 
    Class designed to handle neural network parameters (weights and biases).
    It contains the overloading of operations for Theta objects and few recurrent methods used in the algorithms.
    """
    def __init__(self, theta): 
        self.values = theta

    @property
    def weights(self): 
        """ Extraction of weights from self.values """
        return self.values[0::2]
    @property
    def biases(self): 
        """ Extraction of biases from self.values """
        return self.values[1::2]
    
    def __mul__(self, other): 
        """ Multiplication of each Theta entry by a scalar (int or float) or elementwise multiplication of Thetas """
        if type(other) == int or type(other) == float: return Theta([a*other for a in self.values])
        return Theta([a*b for a,b in zip(self.values, other.values)])
    def __rmul__(self, other): 
        """ Multiplication with the Theta object on the right """
        return self*other
    def __truediv__(self, other) : 
        """ Division of a Theta object by a scalar (int or float) or elementwise division of Thetas"""
        return self*(other**(-1))
    def __rtruediv__(self, other): 
        """ Division with the Theta object on the right """
        return (self**(-1))*other
    
    def __neg__(self): 
        """ Opposite of each Theta entry """
        return self*(-1)
    def __pow__(self, exp): 
        """ Implementation of elementwise square, sqrt and reciprocal of a Theta object """
        if exp == 2:   return Theta([tf.math.square(t)     for t in self.values])
        if exp == 0.5: return Theta([tf.math.sqrt(t)       for t in self.values])
        if exp == -1:  return Theta([tf.math.reciprocal(t) for t in self.values])
    
    def __add__(self, other):
        """ Addition of a scalar (int or float) to each Theta entry or elementwise sum of Thetas """
        if type(other) == int or type(other) == float: return Theta([a+other for a in self.values])
        return Theta([a+b for a,b in zip(self.values, other.values)])
    def __radd__(self, other): 
        """ Addition with the Theta object on the right """
        return self+other
    def __sub__(self, other): 
        """ Subtraction of a scalar (int or float) to each Theta entry or elementwise subtraction of Thetas """
        return self+(-other)
    def __rsub__(self, other): 
        """ Subtraction with the Theta object on the right """
        return -self+other  

    def __len__(self): 
        """ Returns the number of tensors in the self.values list """
        return len(self.values)
    def __str__(self):
        """ Formatted print of a Theta object DA RIFARE?? """
        for i, (w,b) in enumerate(zip(self.weights, self.biases)):
            print(f"W{i}:", w)
            print(f"b{i}:", b)
        return ""

    def exp(self): return Theta([tf.math.exp(t) for t in self.values])
    def log(self): return Theta([tf.math.log(t) for t in self.values])
    
    def ssum(self): 
        """ Squared sum of all entries of self.values """
        return sum([tf.norm(t)**2    for t in self.values])
    def size(self): 
        """ Counter of all entries of self.values """
        return sum([np.prod(t.shape) for t in self.values])
    def copy(self): 
        """ Returns a Theta object with copied self.values """
        return Theta(self.values.copy())
    def normal(self, mean=0.0, std=1.0): 
        """ Creation of a Theta object with random normal initialization of self.values """
        return Theta([tf.random.normal(t.shape, mean=mean, stddev=std) for t in self.values])