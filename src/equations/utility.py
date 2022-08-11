import tensorflow as tf
from abc import ABC, abstractmethod

class Pde_constraint(ABC):
    """
    Parent abstract class for pde constraint
    """
    def __init__(self, par):
        """ Constructor
        n_input   -> dimension input (1,2 or 3)
        n_out_sol -> dimension of solution
        n_out_par -> dimension of parametric field
        """
        self.n_input   = par.n_input
        self.n_out_sol = par.n_out_sol
        self.n_out_par = par.n_out_par

    @abstractmethod
    def compute_pde_residual(self, inputs_pts):
        """compute the pde losses, need to be overridden in child classes"""
        return 0.

    @abstractmethod
    def pre_process(self, dataset):
        return None

    @abstractmethod
    def post_process(self, outputs):
        return None

class Operators:
    
    """
    Class for differential operators
    
    Notations:
    x is the derivation variable (n_sample x dim_input)
    d is dim_output
    s is a scalar function (n_sample x 1)
    v is a vector function (n_sample x dim_output)
    A is a tensor function (n_sample x dim_output x dim_output)
    """
    
    @staticmethod
    def gradient_scalar(tape, s, x):
        """
        input  shape: (n_sample x 1)
        output shape: (n_sample x 1 x dim_input)
        """
        return tf.expand_dims(tape.gradient(s, x), axis=1)

    @staticmethod
    def gradient_vector(tape, v, x, d):
        """
        input  shape: (n_sample x dim_output)
        output shape: (n_sample x dim_output x dim_input)
        """
        return tf.stack([tape.gradient(v[:,i], x) for i in range(d)], axis=-2)
    

    @staticmethod
    def derivate_scalar(tape, s, x):
        """
        input  shape: (n_sample x 1)
        output shape: (n_sample x 1)
        """
        return tf.squeeze(Operators.gradient_scalar(tape, s, x), axis=-1)

    @staticmethod
    def derivate_vector(tape, s, x, d):
        """
        input  shape: (n_sample x dim_output)
        output shape: (n_sample x dim_output)
        """
        return tf.squeeze(Operators.gradient_vector(tape, s, x, d), axis=-1)


    @staticmethod
    def divergence_vector(tape, v, x, d):
        """
        input  shape: (n_sample x dim_output) or
        input  shape: (n_sample x 1 x dim_input)
        output shape: (n_sample x 1)
        """
        if len(v.shape) == 3: v = tf.squeeze(v, axis = -2)
        return tf.expand_dims(tf.linalg.trace(Operators.gradient_vector(tape, v, x, d)), axis=-1)

    @staticmethod
    def divergence_tensor(tape, A, x, d):
        """
        input  shape: (n_sample x dim_output x dim_input)
        output shape: (n_sample x dim_output)
        """
        divergences = [Operators.divergence_vector(tape, A[:,i,:], x, d) for i in range(d)]
        return tf.squeeze(tf.stack(divergences, axis = -1), axis=-2)


    @staticmethod
    def laplacian_scalar(tape, s, x, d):
        """
        input  shape: (n_sample x 1)
        output shape: (n_sample x 1)
        """
        return Operators.divergence_vector(tape, Operators.gradient_scalar(tape, s, x), x, d)

    @staticmethod
    def laplacian_vector(tape, v, x, d):
        """
        input  shape: (n_sample x dim_output)
        output shape: (n_sample x dim_output)
        """
        return Operators.divergence_tensor(tape, Operators.gradient_vector(tape, v, x, d), x, d)
