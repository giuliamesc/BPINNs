import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# tfd = tfp.distributions ## CONTROLLARE

from models.FCN import Net
from models.prior_trainable import trainable_param


class BayesNN:
    """
    Bayesian-PINN: Parent class, inheritance in SVGD_BayesNN and HMC_BayesNN
    """
    def __init__(self, num_neural_networks, sigmas,
                n_input, architecture, n_out_sol, n_out_par, parameters, pde_constr, random_seed):
        """!
        Constructor
        @param num_neural_networks number of neural networks for SVGD, 1 for HMC
        @param sigmas parameters for sigmas
        @param n_input dim of input (1D, 2D or 3D)
        @param architecture nn_architecture parameters
        @param n_out_sol dim of output solution
        @param n_out_par dim of output parametric fields
        @param parameters weights for posterior probability (pde, likelihood and prior)
        @param pde_constr object of pde_constraint class
        """

        ## problem dimensions
        self.n_input = n_input     # n input domain
        self.n_out_sol = n_out_sol # n output solution
        self.n_out_par = n_out_par # n output parametric field


        ## architecture of our Feed Forward Neural Network
        self.nnets = [] # list of neural networks
        self.architecture_nn = []
        self.n_layers = architecture["n_layers"]   # number of hidden layers
        self.n_neurons = architecture["n_neurons"] # number of hidden neurons for each layer

        # append an instance of Net object
        self.nnets.append(Net(self.n_input, architecture["n_layers"],
                              architecture["n_neurons"], self.n_out_sol + self.n_out_par))
        self.architecture_nn = self.nnets[0].get_dimensions()


        # w_i ~ StudentT(w_i | mu=0, lambda=shape/rate, nu=2*shape)
        # (even if we use a classical initializer for NN in FCN.py (zero bias, glorot initialization for W))
        self.w_prior_shape = 1.     # prior_shape of w
        self.w_prior_rate  = 0.05   # prior_rate of w

        # beta ~ Gamma(beta | shape, rate)
        self.beta_prior_shape = 2.                        # beta_prior_shape
        self.beta_prior_rate = sigmas["data_prior_noise"] # beta_prior_rate
        self.beta_pde_shape = 2.                          # beta_pde_shape
        self.beta_pde_rate = sigmas["pde_prior_noise"]    # beta_pde_rate

        # additional trainable parameters
        bool_log_betaD = sigmas["data_prior_noise_trainable"]
        bool_log_betaR = sigmas["pde_prior_noise_trainable"]

        param = {"beta_prior_shape" : self.beta_prior_shape,
                "beta_prior_rate"   : self.beta_prior_rate,
                "beta_pde_shape"    : self.beta_pde_shape,
                "beta_pde_rate"     : self.beta_pde_rate}

        # an object of "trainable_param" class that contains:
        # log_betaD (where log_betaD = log(betaD) = log(1/sigma_D^2), and sigma_D standard deviation of sparse data)
        # log_betaR (where log_betaR = log(betaR) = log(1/sigma_R^2), and sigma_R standard deviation of pde residual in collocation points)
        self.log_betas = trainable_param(bool_log_betaD, bool_log_betaR, param, num_neural_networks, random_seed)

        # store the posterior weights
        self.param_res   = parameters["param_res"]   # weights for pde in posterior
        self.param_data  = parameters["param_data"]  # weights for likelihood in posterior
        self.param_prior = parameters["param_prior"] # weights for prior in posterior

        ## store the pde_constraint
        self.pde_constraint = pde_constr

        ## list to store log losses
        self.res_logloss   = list() # list to store pde log losses
        self.data_logloss  = list() # list to store data log losses
        self.prior_logloss = list() # list to store prior log losses


    def forward(self, inputs):
        """!
        forward pass of inputs data through the neural network
        @param inputs inputs points of shape (input_len, input_dim)
        """
        # compute the output of NN at the inputs data
        output = self.nnets[0].forward(inputs)
        # select solution output
        output_sol = output[:,:self.n_out_sol]
        # select parametric field output
        output_par = output[:,self.n_out_sol:]

        return output_sol, output_par

    def get_trainable_weights(self):
        """Get all the trainable weights of the NN in a list"""
        return self.nnets[0].get_parameters()

    def get_trainable_weights_flatten(self):
        """Get all the trainable weights of the NN in a flatten vector"""
        w = list()
        for param in self.nnets[0].get_parameters():
            w.append(tf.reshape(param,[-1]))
        return tf.convert_to_tensor(tf.concat(w, axis=0))


    # compute the log joint probability = log of data + log_prior (+ log_prior_log_betaD if trainable)
    @tf.function # decorator to speed up the computation
    def log_joint(self, output, target):
        """!
        Log joint probability: log likelihood of sparse exact (noisy) data and prior of weights
        + if trainable, prior of log_betaD

        @param output our prediction of the exact sparse data
        @param target exact sparse data
        """
        # likelihood of exact data:
        # Normal(output | target, 1 / betaD * I)
        mse_data = tf.reduce_mean(tf.keras.losses.MSE(output, target))
        n_d = output.shape[0]
        log_data = (- 0.5 * n_d * mse_data * tf.math.exp(self.log_betas.log_betaD)
                    + 0.5 * n_d * self.log_betas.log_betaD)
        log_data*=self.param_data

        # compute log prior of w (t-student)
        log_prior = 0.
        for param in self.nnets[0].get_parameters():
            log_prior += -0.5*tf.reduce_sum(param**2)
        log_prior*=self.param_prior

        # log prior of a log(inverse gamma)
        if(self.log_betas._bool_log_betaD):
            log_prob_log_betaD = (self.beta_prior_shape-1) * self.log_betas.log_betaD - \
                            self.beta_prior_rate * (tf.math.exp(self.log_betas.log_betaD))
            log_data+=log_prob_log_betaD

        return tf.cast(mse_data,dtype=tf.float32), tf.cast(log_data,dtype=tf.float32), tf.cast(log_prior,dtype=tf.float32)

    # compute the loss and logloss of Physics Constrain (PDE constraint)
    @tf.function # decorator to speed up the computation
    def pde_logloss(self,inputs):
        pass

    def save_networks(self, path):
        """Save networks"""
        pass

    def load_networks(self, path):
        """Load networks"""
        pass

class HMC_BayesNN(BayesNN):
    """
    Define Bayesian-PINN for HMC methods (from BayesNN)
    """
    def __init__(self, sigmas, n_input, architecture, n_out_sol, n_out_par, parameters, pde_constr, random_seed, M):
        """!
        Constructor
        @param sigmas parameters for sigmas
        @param n_input dim of input (1D, 2D or 3D)
        @param architecture nn_architecture parameters
        @param n_out_sol dim of output solution
        @param n_out_par dim of output parametric fields
        @param parameters weights for posterior probability (pde, likelihood and prior)
        @param pde_constr object of pde_constraint class
        @param M param M in HMC (number of samples we want to keep after burn-in period)
        """

        super().__init__(1, sigmas, n_input, architecture, n_out_sol, n_out_par, parameters, pde_constr, random_seed)
        self._thetas = []     ## list to store the thetas
        self._log_betaDs = [] ## list to store log betas D
        self._log_betaRs = [] ## list to store log betas R
        self.M = M  ## number of samples we want to keep after burn-in period

    # compute the loss and logloss of Physics Constrain (PDE constraint)
    @tf.function # decorator to speed up the computation
    def pde_logloss(self,inputs):
        """!
        Compute the loss and logloss of the PDE constraint
        @param inputs tensor of shape (batch_size, n_input)
        """
        # compute loss_1 and loss_2 using pde_constraint
        pde_residual = self.pde_constraint.compute_pde_losses(inputs, self.forward)
        mse_residual = tf.reduce_mean(tf.keras.losses.MSE(pde_residual,tf.zeros_like(pde_residual)))

        # log loss for a Gaussian -> Normal(loss_1 | zeros, 1/betaR*Identity)
        n_r = inputs.shape[0]
        logloss = (- 0.5 * n_r * mse_residual * tf.math.exp(self.log_betas.log_betaR)
                   + 0.5 * n_r * self.log_betas.log_betaR)

        log_loss_total = logloss
        log_loss_total *= self.param_res

        # if log_betaR trainable add his prior (Inv-Gamma)
        if(self.log_betas._bool_log_betaR):
            log_prob_log_betaR = (self.beta_pde_shape-1) * self.log_betas.log_betaR - \
                            self.beta_pde_rate * (tf.math.exp(self.log_betas.log_betaR))
            log_prob_log_betaR *= self.param_prior
            log_loss_total += log_prob_log_betaR

        return tf.cast(mse_residual,dtype=tf.float32), tf.cast(log_loss_total,dtype=tf.float32)


    # save all the weights
    def save_networks(self, path):
        """ Save weights of the neural network in path """
        np.save(os.path.join(path, "thetas.npy"), np.array(self._thetas[-self.M:], dtype=object))
        np.save(os.path.join(path, "architecture_nn.npy"), np.array(self.architecture_nn))
        if(self.log_betas.betas_trainable_flag()):
            if(self.log_betas._bool_log_betaD):
                np.save(os.path.join(path, "log_betaD.npy"), np.array(self._log_betaDs))
            if(self.log_betas._bool_log_betaR):
                np.save(os.path.join(path, "log_betaR.npy"), np.array(self._log_betaRs))

    # load all the neural networks
    def load_networks(self, path):
        """ Load weights of the neural network from path """
        self._thetas = np.load(os.path.join(path, "thetas.npy"), allow_pickle=True)[:self.M]
        if(self.log_betas.betas_trainable_flag()):
            if(self.log_betas._bool_log_betaD):
                self._log_betaDs = np.load(os.path.join(path, "log_betaD.npy"))[:self.M]
            if(self.log_betas._bool_log_betaR):
                self._log_betaRs = np.load(os.path.join(path, "log_betaR.npy"))[:self.M]

    def predict(self, inputs):
        """!
        Predict the output using input=inputs using all the thetas we have stored in self._thetas.
        return two tensors (samples_u and samples_f) of shape:
        samples_u shape = (M, len_inputs, n_out_sol)
        samples_f shape = (M, len_inputs, n_out_par)

        @param inputs inputs data
        """
        samples_u = []
        samples_f = []

        # for loop over the last M thetas we have stored
        # for i in range(len(self._thetas[-self.M:])):
        for i in range(len(self._thetas)):
            # update the weights
            #self.nnets[0].update_weights(self._thetas[-self.M:][i])
            self.nnets[0].update_weights(self._thetas[i])
            # forward pass of inputs
            u_i, f_i = self.forward(inputs)
            # append u_i and f_i
            samples_u.append(u_i)
            samples_f.append(f_i)

        # convert list to numpy tensor
        samples_u = np.array(samples_u)
        samples_f = np.array(samples_f)

        return samples_u, samples_f


    def mean_and_std(self, inputs):
        """!
        Compute mean and std deviation at data_test = inputs
        @param inputs inputs data
        """

        samples_u, samples_f = self.predict(inputs)

        # compute mean and standard deviation
        u_mean = np.mean(samples_u, axis=0)
        f_mean = np.mean(samples_f, axis=0)
        u_std = np.std(samples_u, axis=0)
        f_std = np.std(samples_f, axis=0)

        # add sigma_D or sigma_R if trainable
        sigma_D = 0.
        sigma_R = 0.

        if(self.log_betas._bool_log_betaD):
            b = np.exp(self._log_betaDs)
            s = np.sqrt(np.reciprocal(b))
            sigma_D += np.mean(s)
        if(self.log_betas._bool_log_betaR):
            b = np.exp(self._log_betaRs)
            s = np.sqrt(np.reciprocal(b))
            sigma_R += np.mean(s)

        return u_mean, f_mean, u_std+sigma_D, f_std+sigma_R
