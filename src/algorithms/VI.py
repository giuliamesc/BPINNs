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
        self.alpha = 0.1

    def __update_theta(self):
        eps   = self.VI_rho.normal()
        theta = self.VI_mu + self.VI_rho * eps
        self.nn_params = theta 
        return self.nn_params
    
    def __initialize_VI_params(self):
        theta = self.model.nn_params
        self.VI_mu  = theta.normal(std=1.)*0
        self.VI_rho = theta.normal(std=1.)
        self.__update_theta()

    def __upadte_VI_params(self, grad_mu, grad_rho):
        print("\n")
        print(self.VI_mu.weights[0])
        print(grad_mu.weights[0])
        self.VI_mu  -= grad_mu  * self.alpha
        self.VI_rho -= grad_rho * self.alpha
        print(self.VI_mu.weights[0])
        print("\n")

    def compute_grad_rho(self, grad_theta):
        q_theta   = 0
        q_rho     = 0
        theta_rho = 0
        return (q_theta + grad_theta) * theta_rho + q_rho

    def sample_theta(self):
        grad_theta  = self.model.grad_loss(self.data_batch) 
        grad_rho = self.compute_grad_rho(grad_theta)
        self.__upadte_VI_params(grad_theta, grad_rho)
        return self.__update_theta()

    def select_thetas(self, thetas_train):
        """ Compute burn-in and skip samples """
        return thetas_train