import tensorflow as tf
import numpy as np

class Theta():
    """ 
    Class designed to handle neural network parameters (weights and biases).
    It contains the overloading of operations for Theta objects and few recurrent methods used in the algorithms.
    """
    def __init__(self, theta): 
        self.values = theta
    
    def __mul__(self, other): 
        """ Multiplication of each Theta entry by a scalar (int or float) or elementwise multiplication of Thetas """
        if type(other) == int or type(other) == float: return Theta([a*other for a in self.values])
        return Theta([a*b for a,b in zip(self.values, other.values)])
    def __neg__(self): 
        """ Opposite of each Theta entry """
        return self*(-1)
    def __add__(self, other):
        """ Addition of a scalar (int or float) to each Theta entry or elementwise sum of Thetas """
        if type(other) == int or type(other) == float: return Theta([a+other for a in self.values])
        return Theta([a+b for a,b in zip(self.values, other.values)])
    def __sub__(self, other): 
        """ Subtraction of a scalar (int or float) to each Theta entry or elementwise subtraction of Thetas """
        return self+(-other)
    def __truediv__(self, other) : return self*(other**(-1))
    def __rtruediv__(self, other): return (self**(-1))*other
    def __rmul__(self, other): return self*other
    def __radd__(self, other): return self+other
    
    def __pow__(self, exp): 
        if exp == 2:   return Theta([tf.math.square(t)     for t in self.values])
        if exp == 0.5: return Theta([tf.math.sqrt(t)       for t in self.values])
        if exp == -1:  return Theta([tf.math.reciprocal(t) for t in self.values])

    def __len__(self): return len(self.values)
    
    def __str__(self):
        for i, (w,b) in enumerate(zip(self.weights, self.biases)):
            print(f"W{i}:", w)
            print(f"b{i}:", b)
        return ""
    
    def ssum(self): return sum([tf.norm(t)**2    for t in self.values])
    def size(self): return sum([np.prod(t.shape) for t in self.values])
    def copy(self): return Theta(self.values.copy())
    def normal(self, std): return Theta([tf.random.normal(t.shape, stddev=std) for t in self.values]) 

    @property
    def weights(self): return self.values[0::2]
    @property
    def biases(self): return self.values[1::2]