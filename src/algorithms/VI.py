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
        self.burn_in = param_method["burn_in"]
        self.samples = param_method["VI_samples"]
        self.alpha   = param_method["VI_alpha"]
    
    def __initialize_VI_params(self):
        shape = self.model.nn_params
        self.VI_mu  = shape.normal(std=0.1)
        self.VI_rho = shape.normal(std=0.01)

    def __update_theta(self):
        self.eps = self.VI_rho.normal()
        self.VI_sigma = (self.VI_rho.exp()+1).log()
        self.inv_sigma = self.VI_sigma**(-1) 
        return self.VI_mu + self.VI_sigma * self.eps

    def __compute_grad_rho(self, grad_theta):
        q_theta   = (self.VI_mu - self.model.nn_params) * (self.inv_sigma)**2 # (mu-theta) * 1/sigma^2
        q_rho     = ((self.VI_mu - self.model.nn_params)**2 * (self.inv_sigma)**2 - 1) * (self.inv_sigma) # (mu-theta)^2 * 1/sigma^3 - 1/sigma 
        theta_rho = self.eps / (self.VI_rho.exp()+1) # eps / (1+exp(rho))
        return (q_theta + grad_theta) * theta_rho + q_rho

    def __update_VI_params(self, grad_mu, grad_rho):
        self.VI_mu  -= grad_mu  * self.alpha
        self.VI_rho -= grad_rho * self.alpha

    def sample_theta(self): 
        self.model.nn_params = self.__update_theta()
        grad_theta = self.model.grad_loss(self.data_batch) 
        grad_rho   = self.__compute_grad_rho(grad_theta)
        self.__update_VI_params(grad_theta, grad_rho)
        return self.model.nn_params

    def select_thetas(self, *_):
        """ Compute burn-in and skip samples """
        mu, sigma = (self.VI_mu, (self.VI_rho.exp()+1).log())
        zeta = [sigma.normal() for _ in range(self.samples)]
        return [mu + sigma * z for z in zeta]