import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from FCN import Net
from other_trainable_parameters import trainable_param


class BayesNN:
    """
    Bayesian-PINN: Parent class, inheritance in SVGD_BayesNN and MCMC_BayesNN
    """
    def __init__(self, num_neural_networks, sigmas,
                nFeature, architecture, n_out_sol, n_out_par, parameters, pde_constr, random_seed):
        """!
        Constructor
        @param num_neural_networks number of neural networks for SVGD, 1 for MCMC
        @param sigmas parameters for sigmas
        @param nFeature dim of input (1D, 2D or 3D)
        @param architecture nn_architecture parameters
        @param n_out_sol dim of output solution
        @param n_out_par dim of output parametric fields
        @param parameters weights for posterior probability (pde, likelihood and prior)
        @param pde_constr object of pde_constraint class
        """

        # w_i ~ StudentT(w_i | mu=0, lambda=shape/rate, nu=2*shape)
        # (even if we use a classical initializer for NN in FCN.py (zero bias, glorot initialization for W))
        ## w_prior_shape
        self.w_prior_shape = 1.     # prior_shape of w
        ## w_prior_rate
        self.w_prior_rate = 0.05    # prior_rate of w

        # beta ~ Gamma(beta | shape, rate)
        ## beta_prior_shape
        self.beta_prior_shape = 2.
        ## beta_prior_rate
        self.beta_prior_rate = sigmas["data_prior_noise"]

        ## beta_pde_shape
        self.beta_pde_shape = 2.
        ## beta_pde_rate
        self.beta_pde_rate = sigmas["pde_prior_noise"]

        ## list of neural networks (useful for SVGD)
        self.nnets = []

        ## architecture of our Feed Forward Neural Network
        self.architecture_nn = []

        ## n input
        self.n_input = nFeature

        ## number of hidden layers
        self.n_layers = architecture["n_layers"]

        ## number of hidden neurons for each layer
        self.n_neurons = architecture["n_neurons"]

        ## n output velocity
        self.n_out_sol = n_out_sol
        
        ## n output parametric field
        self.n_out_par = n_out_par

        # append an instance of Net object
        self.nnets.append(Net(self.n_input, architecture["n_layers"],
                              architecture["n_neurons"], self.n_out_sol+ self.n_out_par))
        # N_vel for u + 1 for f
        self.architecture_nn = self.nnets[0].get_dimensions()

        # additional trainable parameters
        bool_log_betaD = sigmas["data_prior_noise_trainable"]
        bool_log_betaR = sigmas["pde_prior_noise_trainable"]

        param = {"beta_prior_shape" : self.beta_prior_shape,
                    "beta_prior_rate" : self.beta_prior_rate,
                    "beta_pde_shape" : self.beta_pde_shape,
                    "beta_pde_rate" : self.beta_pde_rate}


        # add the other parameters:
        # an object of "trainable_param" class that contains:
        # log_betaD (where log_betaD = log(betaD) = log(1/sigma_D^2), and sigma_D standard deviation of sparse data)
        # log_betaR (where log_betaR = log(betaR) = log(1/sigma_R^2), and sigma_R standard deviation of pde residual in collocation points)
        ## other parameters
        self.log_betas = trainable_param(bool_log_betaD, bool_log_betaR,
                                            param, num_neural_networks, random_seed)

        # store the posterior weights
        ## weights for pde in posterior
        self.param_res = parameters["param_res"]
        ## weights for likelihood in posterior
        self.param_data = parameters["param_data"]
        ## weights for prior in posterior
        self.param_prior = parameters["param_prior"]

        ## store the pde_constraint
        self.pde_constraint = pde_constr

        # list to store the log losses

        ## list to store pde log losses
        self.res_logloss = []
        ## list to store data log losses
        self.data_logloss = []
        ## list to store prior log losses
        self.prior_logloss = []
    ###############################################################################################################################


    # get the neural networks of index=idx
    def __getitem__(self, idx):
        """Get the idx neural networks"""
        return self.nnets[idx]


    # forward pass
    def forward(self, inputs):
        """!
        forward pass of inputs data through the neural network
        @param inputs inputs points of shape (input_len, input_dim)
        """

        # compute the output of NN at the inputs data
        output = self.nnets[0].forward(inputs)
        
        # select solution output
        output_sol = output[:,:self.n_out_sol]
        if(len(output_sol.shape) == 1):
            output_sol = tf.expand_dims(output_sol, axis=1)
        
        # select parametric field output
        output_par = output[:,self.n_out_sol:]
        if(len(output_par.shape) == 1):
            output_par = tf.expand_dims(output_par, axis=1)
        
        return output_sol, output_par

    def get_trainable_weights(self):
        """Get all the trainable weights of the NN in a list"""
        weights = []
        weights.append(self.nnets[0].get_parameters())
        return weights

    def get_trainable_weights_flatten(self):
        """Get all the trainable weights of the NN in a flatten vector"""
        w = []
        w_0 = []
        for param in self.nnets[0].get_parameters() :
            w_0.append(tf.reshape(param,[-1]))
        w.append(tf.concat(w_0, axis=0))

        return tf.convert_to_tensor(w)

    # # save all the neural networks
    def save_networks(self, path):
        """Save networks"""
        pass

    # load all the neural networks
    def load_networks(self, path):
        """Load networks"""
        pass

    # compute the log joint probability = loglikelihood of data + log_prior_w (+ log_prior_log_betaD if trainable)
    @tf.function    # decorator @tf.function to speed up the computation
    def log_joint(self, output, target):
        """!
        Log joint probability: log likelihood of sparse exact (noisy) data and prior of weights
        + if trainable, prior of log_betaD

        @param output our prediction of the exact sparse data
        @param target exact sparse data
        """
        # likelihood of exact data:
        # Normal(output | target, 1 / betaD * I)

        loss_d_scalar = 0.  # MSE of exact data
        log_likelihood = [] # compute log likelihood of exact data
        log_likelihood.append(-0.50*tf.math.exp(self.log_betas.log_betaD)*tf.reduce_sum((target[:,0] - output[:,0])**2)
                                + 0.50 * (tf.size(target[:,0], out_type = tf.dtypes.float32)) * self.log_betas.log_betaD)
        loss_d_scalar += tf.keras.losses.MSE(output[:,0], target[:,0])

        # divide by the numerosity of target (number of exact)
        #log_likelihood/= tf.size(target[:,0], out_type = tf.dtypes.float32)
        # multiply by param_data
        log_likelihood = tf.convert_to_tensor(log_likelihood)
        log_likelihood*=self.param_data

        # compute log prior of w (t-student)
        log_prob_prior_w = []
        log_prob_prior_w_0 = 0.
        for param in self.nnets[0].get_parameters():
            #breakpoint()
            log_prob_prior_w_0 += (-0.5*(1.)*tf.reduce_sum(param**2) )#+ 0.50*size*1.)


            #(tf.reduce_sum( tf.math.log1p( 0.5 / self.w_prior_rate * (param**2) ) ) )
        #log_prob_prior_w_0 *= -(self.w_prior_shape + 0.5)

        log_prob_prior_w.append(log_prob_prior_w_0)
        log_prob_prior_w = tf.convert_to_tensor(log_prob_prior_w)
        # divide by numerosity of neurons ( ~ hidden_layers * n_neurons)
        #log_prob_prior_w /= tf.dtypes.cast(self.n_layers*self.n_neurons, dtype=tf.float32)
        # multiply by param_prior
        log_prob_prior_w*=self.param_prior

        # if log_betaD trainable -> add prior of log_betaD
        if(self.log_betas._bool_log_betaD):
            # log prior of a log(inverse gamma)
            log_prob_log_betaD = (self.beta_prior_shape-1) * self.log_betas.log_betaD - \
                            self.beta_prior_rate * (tf.math.exp(self.log_betas.log_betaD))
            log_likelihood+=log_prob_log_betaD
        # compute the sum of everything (log_likelihood of exact data + log prior w + log prior of log_betaD)
        log_likelihood_total = log_likelihood + log_prob_prior_w

        return log_likelihood_total, loss_d_scalar, log_likelihood,log_prob_prior_w

    # compute the derivative of at and v wrt x and y
    @tf.function    # decorator @tf.function to speed up the computation
    def _gradients(self,inputs):
        pass

    # compute the loss and logloss of Physics Constrain (PDE constraint)
    @tf.function # decorator @tf.function to speed up the computation
    def pde_logloss(self,inputs):
        """!
        Compute the loss and logloss of the PDE constraint

        @param inputs tensor of shape (batch_size, n_input) a single batch of the collocation points """

        # compute the derivatives
        at_gr, v_gr, v = self._gradients(inputs)

        # compute loss_1 and loss_2 using pde_constraint
        loss_1 = self.pde_constraint.compute_pde_losses(at_gr, v_gr, v)

        # compute loss_1_scalar and loss_2_scalar
        loss_1_scalar = tf.keras.losses.MSE(loss_1,tf.zeros_like(loss_1)) #shape (1,) for HMC; (num_neural_networks,) for SVGD

        # likelihood of pde constraints:

        # Normal(loss_1 | zeros, 1/betaR*Identity)
        # log loss for a Gaussian
        logloss1 = (- 0.5 * tf.math.exp(self.log_betas.log_betaR) *
                    tf.reduce_sum((loss_1 - tf.zeros_like(loss_1))**2, axis = 0)
                    + 0.50 * (tf.size(inputs[:,0], out_type = tf.dtypes.float32))
                    * self.log_betas.log_betaR)

        log_loss_total = logloss1
        # multiply by param_res
        log_loss_total*= self.param_res

        # if log_betaR trainable add his prior (Inv-Gamma)
        if(self.log_betas._bool_log_betaR):
            log_prob_log_betaR = (self.beta_pde_shape-1) * self.log_betas.log_betaR - \
                            self.beta_pde_rate * (tf.math.exp(self.log_betas.log_betaR))
            log_loss_total+=log_prob_log_betaR

        # compute the mean losses
        loss_1_scalar = tf.reduce_mean(loss_1_scalar)

        return log_loss_total, loss_1_scalar

class MCMC_BayesNN(BayesNN):
    """
    Define Bayesian-PINN for MCMC methods (from BayesNN)
    """
    def __init__(self, sigmas, nFeature, architecture, n_out_sol, n_out_par, parameters, pde_constr, random_seed, M):
        """!
        Constructor
        @param sigmas parameters for sigmas
        @param nFeature dim of input (1D, 2D or 3D)
        @param architecture nn_architecture parameters
        @param n_out_sol dim of output solution
        @param n_out_par dim of output parametric fields
        @param parameters weights for posterior probability (pde, likelihood and prior)
        @param pde_constr object of pde_constraint class
        @param M param M in HMC (number of samples we want to keep after burn-in period)
        """

        super().__init__(1, sigmas, nFeature, architecture, n_out_sol, n_out_par, parameters, pde_constr, random_seed)

        ## list to store the thetas
        self._thetas = []
        ## list to store log betas D
        self._log_betaDs = []
        ## list to store log betas R
        self._log_betaRs = []

        ## (number of samples we want to keep after burn-in period) (useful for save networks method)
        self.M = M

    ###############################################################################################################################

    # compute the derivative of at and v wrt x and y
    @tf.function    # decorator @tf.function to speed up the computation
    def _gradients(self,inputs):
        """!
        Compute the gradients of at and v wrt to inputs x, y and z
        in (a batch of) collocation points.

        @param inputs tensor of shape (batch_size, n_input) a single batch of the collocation points
        """
        # x,y,z inputs
        x = inputs[:,0:1]
        if(self.n_input>1):
            y = inputs[:,1:2]
            if(self.n_input == 3):
                z = inputs[:,2:3]

        with tf.GradientTape(persistent = True) as t1:
            # watch the inputs
            t1.watch(x)
            if(self.n_input>1):
                t1.watch(y)
                if(self.n_input == 3):
                    t1.watch(z)
                    inputs = tf.concat( (x,y,z), axis=1 )
                else:
                    inputs = tf.concat((x,y), axis=1)
            else:
                inputs = x
            # compute the output by forward pass of inputs
            at,v = self.forward(inputs) #output after forward(input)

            at_x = t1.gradient(at, x)   # dT/dx
            if(self.n_input > 1):
                at_y = t1.gradient(at, y) #dT/dy
                if(self.n_input == 3):
                    at_z = t1.gradient(at, z)   #dT/dz

            #breakpoint()
            # store the gradients
            at_gradients = []
            at_xx = t1.gradient(at_x,x)
            at_gradients.append(at_xx)
            if(self.n_input > 1):
                at_yy = t1.gradient(at_y,y)
                at_gradients.append(at_yy)
                if(self.n_input == 3):
                    at_zz = t1.gradient(at_z,z)
                    at_gradients.append(at_zz)




        del t1
        return at_gradients,v

    # compute the loss and logloss of Physics Constrain (PDE constraint)
    @tf.function # decorator @tf.function to speed up the computation
    def pde_logloss(self,inputs):
        """!
        Compute the loss and logloss of the PDE constraint

        @param inputs tensor of shape (batch_size, n_input) a single batch of the collocation points """

        # compute the derivatives
        at_gr_2, v = self._gradients(inputs)

        # compute loss_1 and loss_2 using pde_constraint
        loss_1 = self.pde_constraint.compute_pde_losses(at_gr_2, v)

        # compute loss_1_scalar and loss_2_scalar
        loss_1_scalar = tf.keras.losses.MSE(loss_1,tf.zeros_like(loss_1)) #shape (1,) for HMC; (num_neural_networks,) for SVGD

        # likelihood of pde constraints:

        # Normal(loss_1 | zeros, 1/betaR*Identity)
        # log loss for a Gaussian
        logloss1 = (- 0.5 * tf.math.exp(self.log_betas.log_betaR) *
                    tf.reduce_sum((loss_1 - tf.zeros_like(loss_1))**2, axis = 0)
                    + 0.50 * (tf.size(inputs[:,0], out_type = tf.dtypes.float32))
                    * self.log_betas.log_betaR)

        log_loss_total = logloss1
        # multiply by param_res
        log_loss_total*= self.param_res

        # if log_betaR trainable add his prior (Inv-Gamma)
        if(self.log_betas._bool_log_betaR):
            log_prob_log_betaR = (self.beta_pde_shape-1) * self.log_betas.log_betaR - \
                            self.beta_pde_rate * (tf.math.exp(self.log_betas.log_betaR))
            log_loss_total+=log_prob_log_betaR

        # compute the mean losses
        loss_1_scalar = tf.reduce_mean(loss_1_scalar)

        return log_loss_total, loss_1_scalar


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
        return two tensors (samples_at and samples_v) of shape:
        samples_at shape = (M, len_inputs, 1)
        samples_v shape = (M, len_inputs, n_out_par)

        @param inputs inputs data
        """
        if(len(self._thetas)==0):
            print("You need to train the model before")
        else:
            samples_at = []
            samples_v = []

            # for loop over the last M thetas we have stored
            #for i in range(len(self._thetas[-self.M:])):
            for i in range(len(self._thetas)):
                # update the weights
                #self.nnets[0].update_weights(self._thetas[-self.M:][i])
                self.nnets[0].update_weights(self._thetas[i])
                # forward pass of inputs
                at_i,v_i = self.forward(inputs)
                # append at_i and v_i
                samples_at.append(at_i)
                samples_v.append(v_i)

            # convert list to numpy tensor
            samples_at = np.array(samples_at)
            samples_v = np.array(samples_v)
            return samples_at, samples_v


    def mean_and_std(self, inputs):
        """!
        Compute mean and std deviation at data_test = inputs

        @param inputs inputs data
        """
        if(len(self._thetas)==0):
            print("You need to train the model before")
        else:
            samples_at = []
            samples_v = []

            # for loop over the last M thetas we have stored
            #for i in range(len(self._thetas[-self.M:])):
            for i in range(len(self._thetas)):
                # update the weights
                #self.nnets[0].update_weights(self._thetas[-self.M:][i])
                self.nnets[0].update_weights(self._thetas[i])
                # forward pass of inputs
                at_i,v_i = self.forward(inputs)
                # append at_i and v_i
                samples_at.append(at_i)
                samples_v.append(v_i)

            # convert list to numpy tensor
            samples_at = np.array(samples_at)
            samples_v = np.array(samples_v)

            # compute mean and standard deviation
            at_mean = np.mean(samples_at, axis=0)
            v_mean = np.mean(samples_v, axis=0)
            at_std = np.std(samples_at, axis=0)
            v_std = np.std(samples_v, axis=0)

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

            return at_mean,v_mean, at_std+sigma_D, v_std+sigma_R


class SVGD_BayesNN(BayesNN):
    """
    Define Bayesian-PINN for SVGD methods (from BayesNN)
    """
    def __init__(self, num_neural_networks, sigmas, nFeature, architecture, n_out_sol, n_out_par, parameters, pde_constr, random_seed):
        """!
        Constructor
        @param num_neural_networks number of neural networks ("number of particles") we want to use to approx the posterior distributions
        @param sigmas parameters for sigmas
        @param nFeature dim of input (1D, 2D or 3D)
        @param architecture nn_architecture parameters
        @param n_out_sol dim of output solution
        @param n_out_par dim of parametric field
        @param parameters weights for posterior probability (pde, likelihood and prior)
        @param pde_constr object of pde_constraint class
        """

        super().__init__(num_neural_networks, sigmas, nFeature, architecture, n_out_sol, n_out_par, parameters, pde_constr, random_seed)

        ## num num_neural_networks
        self.num_neural_networks = num_neural_networks

        # add the others (num_neural_networks-1) NNs at the list self.nnets
        for i in range(self.num_neural_networks-1):
            new_instance = Net(nFeature, architecture["n_layers"],
                            architecture["n_neurons"], self.n_out_par+self.n_out_sol)

            self.nnets.append(new_instance) # append the i-th NN



    ###############################################################################################################################

    # forward pass of all the n_samples neural networks
    def forward(self, inputs):
        """!

        Forward pass through all the num_neural_networks NNs of inputs data

        @param inputs inputs points of shape (input_len, input_dim)"""
        output = []

        # for loop over the num_neural_networks NNs
        for i in range(self.num_neural_networks):
            # append the output of each NN
            output.append(self.nnets[i].forward(inputs))

        output = tf.squeeze(tf.stack(output,axis=1))

        output_sol = output[:,:,:self.n_out_sol]    # shape = (input_len, num_neural_networks, n_out_sol)
        output_par = output[:,:,self.n_out_sol:]    # shape = (input_len, num_neural_networks, n_out_par)

        return output_sol, output_par

    # used only in _gradients()
    def _forward_stacked(self, x_stacked, y_stacked=None, z_stacked=None):
        """Forward pass through all the NNs at the same time. Useful to compute the derivatives wrt inputs in _gradients"""
        output = []
        for i in range(self.num_neural_networks):
            if(self.n_input == 3):
                inputs = tf.concat((x_stacked[:,i:(i+1)],y_stacked[:,i:(i+1)], z_stacked[:,i:(i+1)]),1)
            elif(self.n_input == 2):
                inputs = tf.concat((x_stacked[:,i:(i+1)],y_stacked[:,i:(i+1)]),1)
            else:
                inputs = x_stacked[:,i:(i+1)]

            ## append to output the forward pass of inputs through the i-th neural network
            output.append(self.nnets[i].forward(inputs))

        ## return act. times and velocity separately

        output = tf.squeeze(tf.stack(output,axis=1))
        output_sol = output[:,:,:self.n_out_sol]
        output_par = output[:,:,self.n_out_sol:]

        return output_sol, output_par

    def get_trainable_weights(self):
        """Get all the trainable weights of all the num_neural_networks NNs in a list"""
        weights = []
        for i in range(self.num_neural_networks):
            weights.append(self.nnets[i].get_parameters())
        return weights

    def get_trainable_weights_flatten(self):
        """Get all the trainable weights of all the num_neural_networks NNs in a tensor of shape (num_neural_networks, dim_weights)"""
        w = []
        for i in range(self.num_neural_networks):
            w_i = []
            for param in self.nnets[i].get_parameters() :
                w_i.append(tf.reshape(param,[-1]))
            w.append(tf.concat(w_i, axis=0))

        return tf.convert_to_tensor(w)

    ## save all the neural networks
    def save_networks(self, path):
        """Save the weights of all the num_neural_networks NNs in path"""
        for i in range(self.num_neural_networks):
            save_path = os.path.join(path, "weights_"+str(i)+".h5")
            self.nnets[i]._features.save_weights(save_path)

    # load all the neural networks
    def load_networks(self, path):
        """Load the weights of all the num_neural_networks NNs from path"""
        for i in range(self.num_neural_networks):
            load_path = os.path.join(path, "weights_"+str(i)+".h5")
            self.nnets[i]._features.load_weights(load_path)


    # compute the log joint probability = loglikelihood + log_prior_w + log_prior_log_beta
    @tf.function    # decorator @tf.function to speed up the computation
    def log_joint(self, output, target):
        """!
        Log joint probability: log likelihood of sparse exact (noisy) data and prior of weights

        @param output our prediction of the exact sparse data
        @param target exact sparse data
        """
        # likelihood of exact data:
        # Normal(output | target, 1 / betaD * I)

        # compute the MSE of exact data (for every neural network)
        loss_d_scalar = 0.

        # compute log likelihood of exact data for every neural network
        log_likelihood = []
        for i in range(self.num_neural_networks):
            log_likelihood.append(-0.50*tf.math.exp(self.log_betas.log_betaD[i])*tf.reduce_sum((target[:,0] - output[:,i])**2)
                                + 0.50 * (tf.size(target[:,0], out_type = tf.dtypes.float32)) * self.log_betas.log_betaD[i])
            # add for every neural network the MSE between output[i] and target
            loss_d_scalar += tf.keras.losses.MSE(output[:,i], target[:,0])


        # convert the list to a tensor
        log_likelihood = tf.convert_to_tensor(log_likelihood, dtype=tf.float32)
        # divide by size of target (number of exact)
        #log_likelihood/= tf.size(target[:,0], out_type = tf.dtypes.float32)
        # multiply by param_data
        log_likelihood*=self.param_data

        # divide loss d by num num_neural_networks
        loss_d_scalar /= self.num_neural_networks

        # compute log prior of w (t-student)
        log_prob_prior_w = []
        for i in range(self.num_neural_networks):
            # log prob of prior of weights, i.e. log prob of studentT
            log_prob_prior_w_i = 0.
            for param in self.nnets[i].get_parameters():
                log_prob_prior_w_i += (-0.5*(1.)*tf.reduce_sum(param**2) )
            log_prob_prior_w.append(log_prob_prior_w_i)

        # convert to tensor
        log_prob_prior_w = tf.convert_to_tensor(log_prob_prior_w, dtype=tf.float32)
        # divide by numerosity of neurons ( ~ hidden_layers * n_neurons)
        #log_prob_prior_w /= tf.dtypes.cast(self.n_layers*self.n_neurons, dtype=tf.float32)
        # multiply by param prior
        log_prob_prior_w*=self.param_prior

        # if log_betaD trainable add his prior (inverse gamma)
        if(self.log_betas._bool_log_betaD):
            # log prob of a log(inv gamma)
            log_prob_log_betaD = (self.beta_prior_shape-1) * self.log_betas.log_betaD - \
                            self.beta_prior_rate * (tf.math.exp(self.log_betas.log_betaD))
            log_prob_log_betaD = tf.dtypes.cast(log_prob_log_betaD, dtype=tf.float64)
            log_likelihood+=log_prob_log_betaD

        # compute the sum of everything (log_likelihood of exact data + log prior w + log prior of log_betaD)
        log_likelihood_total = log_likelihood + log_prob_prior_w

        return log_likelihood_total, loss_d_scalar, log_likelihood,log_prob_prior_w

    # compute the derivative of at and v wrt x and y
    @tf.function    # decorator @tf.function to speed up the computation
    def _gradients(self,inputs):
        """!
        Compute the gradients of at and v wrt to inputs x, y and z
        in (a batch of) collocation points.

        @param inputs tensor of shape (batch_size, n_input) a single batch of the collocation points """

        ########################################################################
        # we need these passages to be able to compute derivative with respect
        # of every neural networks at the same time
        ########################################################################

        # repeat x (shape = 100,1) -> num_neural_networks times -> shape=100,30
        xx = tf.repeat(inputs[:,0:1], self.num_neural_networks, axis=1) #100x30
        if(self.n_input>1):
            yy = tf.repeat(inputs[:,1:2], self.num_neural_networks, axis=1)
            if(self.n_input == 3):
                zz = tf.repeat(inputs[:,2:3], self.num_neural_networks, axis=1)

        # GradientTape
        with tf.GradientTape(persistent = True) as t2:
            t2.watch(xx)
            if(self.n_input>1):
                t2.watch(yy)
                if(self.n_input == 3):
                    t2.watch(zz)
            with tf.GradientTape(persistent = True) as t1:
                # watch the inputs
                t1.watch(xx)
                if(self.n_input>1):
                    t1.watch(yy)
                    if(self.n_input == 3):
                        t1.watch(zz)
                        at,v = self._forward_stacked(xx, yy, zz)
                    else:
                        at,v = self._forward_stacked(xx, yy)
                else:
                    # we use _forward_stacked to be able to compute derivative of every neural network at the same time
                    at,v = self._forward_stacked(xx)

            at_x = t1.gradient(at, xx)  # dT/dx (for every neural networks, since xx now has shape (batches_size,num_neural_networks))
            if(self.n_input > 1):
                at_y = t1.gradient(at, yy)  # dT/dy
                if(self.n_input == 3):
                    at_z = t1.gradient(at, zz) # dT/dz

        # store the gradientsèà
        at_gradients = []

        at_xx = t2.gradient(at_x, xx)  # dT/dx (for every neural networks, since xx now has shape (batches_size,num_neural_networks))
        at_gradients.append(at_xx)

        if(self.n_input > 1):
            at_yy = t2.gradient(at_y, yy)  # dT/dy
            at_gradients.append(at_yy)
            if(self.n_input == 3):
                at_zz = t2.gradient(at_z, zz) # dT/dz
                at_gradients.append(at_zz)

        del t1
        del t2
        return at_gradients,v


    # compute the loss and logloss of Physics Constrain (PDE constraint)
    @tf.function # decorator @tf.function to speed up the computation
    def pde_logloss(self,inputs):
        """!
        Compute the loss and logloss of the PDE constraint

        @param inputs tensor of shape (batch_size, n_input) a single batch of the collocation points """

        # compute the derivatives
        at_gr_2, v = self._gradients(inputs)

        # compute loss_1 and loss_2 using pde_constraint
        loss_1 = self.pde_constraint.compute_pde_losses(at_gr_2, v)

        # compute loss_1_scalar and loss_2_scalar
        loss_1_scalar = tf.keras.losses.MSE(loss_1,tf.zeros_like(loss_1)) #shape (1,) for HMC; (num_neural_networks,) for SVGD

        # likelihood of pde constraints:

        # Normal(loss_1 | zeros, 1/betaR*Identity)
        # log loss for a Gaussian
        logloss1 = (- 0.5 * tf.math.exp(self.log_betas.log_betaR) *
                    tf.reduce_sum((loss_1 - tf.zeros_like(loss_1))**2, axis = 0)
                    + 0.50 * (tf.size(inputs[:,0], out_type = tf.dtypes.float32))
                    * self.log_betas.log_betaR)

        log_loss_total = logloss1
        # multiply by param_res
        log_loss_total*= self.param_res

        # if log_betaR trainable add his prior (Inv-Gamma)
        if(self.log_betas._bool_log_betaR):
            log_prob_log_betaR = (self.beta_pde_shape-1) * self.log_betas.log_betaR - \
                            self.beta_pde_rate * (tf.math.exp(self.log_betas.log_betaR))
            log_loss_total+=log_prob_log_betaR

        # compute the mean losses
        loss_1_scalar = tf.reduce_mean(loss_1_scalar)

        return log_loss_total, loss_1_scalar


    def predict(self, inputs):
        """!
        Predict the output using input=inputs using all the num_neural_networks NNs.
        return two tensors (samples_at and samples_v) of shape:
        samples_at shape = (len_inputs, num_neural_networks)
        samples_v shape = (len_inputs, n_out_par, n_out_par)

        @param inputs inputs data
        """
        # forward pass of inputs
        y_time, y_veloc = self.forward(inputs)
        return y_time, y_veloc

    def mean_and_std(self, inputs):
        """!
        Compute mean and std deviation at data_test = inputs

        @param inputs inputs data
        """
        y_time, y_veloc = self.predict(inputs)

        # compute the mean of every NNs
        y_time_mean = tf.math.reduce_mean(y_time, axis=1, keepdims=True).numpy()
        y_veloc_mean = tf.math.reduce_mean(y_veloc, axis=1, keepdims=True).numpy()

        # compute the std of every NNs
        y_time_std = tf.math.reduce_std(y_time, axis=1, keepdims=True).numpy()
        y_veloc_std = tf.math.reduce_std(y_veloc, axis=1, keepdims=True).numpy()

        # compute sigma_D and sigma_R additional if trainable
        sigma_D = 0.
        sigma_R = 0.
        if(self.log_betas._bool_log_betaD):
            b = np.exp(self.log_betaD.numpy())
            s = np.sqrt(np.reciprocal(b))
            sigma_D += np.mean(s)
        if(self.log_betas._bool_log_betaR):
            b = np.exp(self.log_betaR.numpy())
            s = np.sqrt(np.reciprocal(b))
            sigma_R += np.mean(s)

        return y_time_mean, y_veloc_mean, y_time_std+sigma_R, y_veloc_std+sigma_R
