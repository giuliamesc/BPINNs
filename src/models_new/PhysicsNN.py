import tensorflow as tf
from abc import ABC, abstractmethod
from Operators import *

class PhysicsNN():
    
    def __init__(self, par, dataset, model):
        self.par = par
        self.model = model
        self.col_data = dataset.coll_data
        self.fit_data = dataset.exact_data_noise
        self.equation = self.build_equation(self.par.pde)

    def build_equation(self, name_equation):
        if name_equation == "laplace": 
            return laplace(self.col_data[0], self.model.forward, self.par)
        else: assert("This equation is not implemented")

    def loss_total(self):
        loss = dict()
        logloss = dict()
        loss["res"],   logloss["res"]   = self.loss_res(self.col_data[0])
        loss["data"],  logloss["data"]  = self.loss_data(self.fit_data[1], self.fit_data[2])
        loss["prior"], logloss["prior"] = self.loss_prior()
        loss["Total"], logloss["Total"] = sum(loss.values()), sum(logloss.values())
        return loss, logloss

    @tf.function # decorator to speed up the computation
    def loss_residual(self):
        """
        Compute the loss and logloss of the pde
        AGGIUNGI DIMENSIONI
        """
        # compute loss using pde_constraint
        pde_res = self.equation.compute_pde_residual()
        mse_res = tf.reduce_mean(tf.keras.losses.MSE(pde_res, tf.zeros_like(pde_res)))

        # log loss for a Gaussian -> Normal(loss_1 | zeros, 1/betaR*Identity)
        n_r = pde_res.shape[0] # number of samples
        log_var = self.par.sigmas["pde_prior_noise"] # log(1/betaR)

        log_res = self.normal_loglikelihood(mse_res, n_r, log_var)
        log_res *= self.par.param_res

        return self.convert(mse_res), self.convert(log_res)

    @tf.function # decorator to speed up the computation
    def loss_data(self, outputs, targets):
        """
        Compute the loss and logloss of the data 
        AGGIUNGI DIMENSIONI
        """
        # Normal(output | target, 1 / betaD * I)
        mse_data = tf.reduce_mean(tf.keras.losses.MSE(outputs, targets))

        n_d = outputs.shape[0]
        log_var = self.par.sigmas["data_prior_noise"] # log(1/betaD)

        log_data = self.normal_loglikelihood(mse_data, n_d, log_var)
        log_data*=self.par.param_data

        return self.convert(mse_data), self.convert(log_data)

    @tf.function # decorator to speed up the computation
    def loss_prior(self):
        """
        Compute the logloss of the prior 
        AGGIUNGI DIMENSIONI
        """
        loss_prior = 0.
        log_prior = 0.
        # compute log prior of w (t-student)
        log_prior *= self.par.param_prior
        return loss_prior, log_prior

    @staticmethod
    def convert(tensor): 
        return tf.cast(tensor, dtype=tf.float32)

    @staticmethod
    def normal_loglikelihood(mse, n, log_var):
        return (- 0.5 * n * mse * tf.math.exp(log_var) + 0.5 * n * log_var)

class pde_constraint(ABC):
    """
    Class for the pde_constraint
    """
    def __init__(self, inputs_pts, forward_pass, par):
        """ Constructor
        n_input   -> dimension input (1,2 or 3)
        n_out_sol -> dimension of solution
        n_out_par -> dimension of parametric field
        """
        self.forward   = forward_pass
        self.input_pts = inputs_pts
        self.n_input   = par.n_input
        self.n_out_sol = par.n_out_sol
        self.n_out_par = par.n_out_par

    @abstractmethod
    def compute_pde_residual(self):
        """compute the pde losses, need to be overridden in child classes"""
        return 0.
    

class laplace(pde_constraint):
    def __init__(self, inputs_pts, forward_pass, par):
        super().__init__(inputs_pts, forward_pass, par)
        
    def compute_pde_residual(self):
        """
        - Laplacian(u) = f -> f + Laplacian(u) = 0
        u shape: (n_sample x n_out_sol)
        f shape: (n_sample x n_out_par)
        """
        x = self.input_pts
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u, f = self.forward_pass(x)
            lap = Operators.laplacian_vector(tape, u, x, self.n_out_sol)
        return lap + f
