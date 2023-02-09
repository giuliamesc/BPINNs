import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm

class VI(Algorithm):
    """
    Class for VI training
    """
    def __init__(self, bayes_nn, param_method, debug_flag):
        super().__init__(bayes_nn, param_method, debug_flag)
        self.__initialize_VI_params()
        self.alpha = 1e-3
    
    def __initialize_VI_params(self):
        shape = self.model.nn_params
        self.VI_mu  = shape.normal(std=0.)
        self.VI_rho = shape.normal(std=0.01)

    def __update_theta(self):
        self.eps = self.VI_rho.normal()
        self.inv_sigma = (self.VI_rho.exp()+1).log()**(-1) 
        return self.VI_mu + self.VI_rho * self.eps

    def __compute_grad_rho(self, grad_theta):
        q_theta   = (self.VI_mu - self.model.nn_params) * (self.inv_sigma)**2 # (mu-theta) * 1/sigma^2
        q_rho     = ((self.VI_mu - self.model.nn_params)**2 * (self.inv_sigma)**2 - 1) * (self.inv_sigma) # (mu-theta)^2 * 1/sigma^3 - 1/sigma 
        theta_rho = self.eps / (self.VI_rho.exp()+1) # eps / (1+exp(rho))
        return (q_theta + grad_theta) * theta_rho + q_rho

    def __upadte_VI_params(self, grad_mu, grad_rho):
        print("\n")
        print(self.VI_mu.weights[0])
        print(grad_mu.weights[0])

        self.VI_mu  -= grad_mu  * self.alpha
        self.VI_rho -= grad_rho * self.alpha
        
        print(self.VI_mu.weights[0])
        print("\n")

    def sample_theta(self):
        self.model.nn_params = self.__update_theta()
        grad_theta = self.model.grad_loss(self.data_batch) 
        grad_rho   = self.__compute_grad_rho(grad_theta)
        self.__upadte_VI_params(grad_theta, grad_rho)
        return self.model.nn_params

    def select_thetas(self, thetas_train):
        """ Compute burn-in and skip samples """
        return thetas_train