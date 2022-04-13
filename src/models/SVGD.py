# module import
import tensorflow as tf
import tensorflow_probability as tfp

# standard import
import time
import math
import os
from tqdm import tqdm

### Stein Variational Gradient Descend
class SVGD:
    """
    SVGD (Stein Variational Gradient Descend) method
    """

    def __init__(self, bayes_nn,train_loader,datasets_class, parameters):
        """!
        constructor

        @param bayes_nn an object of type SVGD_BayesNN that collects all the num_neural_networks NNs and all the methods to compute the posterior
        @param train_loader the data loader in minibatch for the collocation points
        @param datasets_class
        @param parameters parameter for the SVGD method (in json file, "SVGD" section)
        """

        ## BayesNN
        self.bayes_nn = bayes_nn
        ## Dataloader in batch of collocation data after shuffle
        self.train_loader = train_loader
        ## datasets_class, store all the datasets (collocation, exact, exact_with_noise etc.)
        self.datasets_class = datasets_class

        ## number of neural networks (num_neural_networks)
        self.num_neural_networks = parameters["n_samples"]
        ## learning rate for weights in the NNs
        self.lr = parameters["lr"]
        ## learning rate for log_betaD and log_betaR if they are trainable
        self.lr_noise = parameters["lr_noise"]
        ## number of epochs
        self.epochs = parameters["epochs"]

        ## store the optimizers (Adam) of both weights (lr) and log_betas (lr_noise)
        self.optimizers = self._optimizers(self.lr, self.lr_noise)  # store the optimizers

        ## input dimension
        self.n_input = self.bayes_nn.n_input
        ## output dimension of solution
        self.n_out_sol = self.bayes_nn.n_out_sol
        ## output dimension of parametric field
        self.n_out_par = self.bayes_nn.n_out_par

        ## parameter of repulsivity between the num_neural_networks different NNs
        self.param_repulsivity = tf.constant(parameters["param_repulsivity"],
                                            dtype=tf.float32)

    def _squared_dist(self, X):
        """!
        Computes squared distance between each row of `X`, ||X_i - X_j||^2
        arg: X (Tensor): (S, P) where S is num_neural_networks, P is the dim of
                one sample
        Returns:
            (Tensor) (S, S)

        @param X tensor of dim (num_neural_networks, dim_of_theta)
        """
        XXT = tf.linalg.matmul(X, tf.transpose(X))  ### SxS
        XTX = tf.linalg.diag_part(XXT)
        return -2.0 * XXT + XTX + tf.expand_dims(XTX, 1)


    def _Kxx_dxKxx(self, X):

        """
        Computes covariance matrix K(X,X) and its gradient w.r.t. X
        for RBF kernel with design matrix X, as in the second term in eqn (8)
        of reference SVGD paper.
        Args:
            X (Tensor): (S, P), design matrix of samples, where S is num of
                samples, P is the dim of each sample which stacks all params
                into a (1, P) row.
        we compute Kxx and dxKxx, the last using also the param_repulsivity

        @param X tensor
        """
        # compute the squared distance
        squared_dist = self._squared_dist(X)
        # compute the median
        median = tfp.stats.percentile(squared_dist, 50.0, interpolation='midpoint')
        # compute the consant l for RBF kernel
        l_square = 0.5 * median / math.log(self.num_neural_networks)
        # compute Kxx
        Kxx = tf.math.exp(-0.5 / l_square * squared_dist)

        # compute the derivative of Kxx
        dxKxx = tf.linalg.matmul( (tf.linalg.diag(tf.math.reduce_sum(Kxx, axis=1)) - Kxx  ), X) / l_square
        # multiply for the repulsivity parameter
        dxKxx *= self.param_repulsivity

        return Kxx, dxKxx


    def _optimizers(self, lr, lr_noise):
        """!
        Initialize Adam optimizers (first for theta(lr), second for log_beta(lr_noise))
        @param lr learning rate for NN parameters theta
        @param lr_noise learning rate for log_betas
        """
        optimizers = []

        optimizer_theta = tf.keras.optimizers.Adam(learning_rate=lr)
        optimizers.append(optimizer_theta)

        optimizer_log_beta = tf.keras.optimizers.Adam(learning_rate=lr_noise)
        optimizers.append(optimizer_log_beta)
        return optimizers


    # backpropagation -> compute the gradients of the parameters of NN and of log_beta for index=i
    @tf.function()    # decorator @tf.function to speed up the computation
    def _compute_backprop_gradients(self, sp_inputs, sp_target, inputs):
        """!
        Compute gradient of theta (for each one of the num_neural_networks NNs)
        through backpropagation (using autodifferentiation -> tf.GradientTape)
        @param sp_inputs inputs of noisy exact data, shape=(n_exact, n_input)
        @param sp_target activation times of noisy exact data, shape=(n_exact, 1)
        @param inputs inputs of collocation data (batch), shape=(batch_size, n_input)
        """
        # Automatic Differentiation
        ## get all the trainable weights (for each neural networks)
        param = self.bayes_nn.get_trainable_weights()
        ## flag if we want to train also on at least one log_beta
        flag = self.bayes_nn.log_betas.betas_trainable_flag()

        ## if flag is True, get trainable log_betas
        if(flag):
            betas_trainable = self.bayes_nn.log_betas.get_trainable_log_betas()

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(param) # enable derivative wrt all the self.bayes_nn[i].features.trainable_weights (parameters) (for each i=1,...,num_neural_networks)
            if(flag):
                tape.watch(betas_trainable) # if flag=True, enable derivative also wrt betas_trainable

            ## forward prop for exact data for all the neural networks (we need just the act.times)
            sp_output_times,_ = self.bayes_nn.forward(sp_inputs) #shape = (n_exact, num_neural_networks)

            ## logeq and losses for batch collocation points (PDE constraints)
            log_eq, loss_1,loss_2 =  self.bayes_nn.pde_logloss(inputs)

            ## logjoint (log_likelihood+log_prior_w) for exact (noisy) points (Exact data constraint)
            log_joint, loss_d, log_likelihood,log_prior_w = self.bayes_nn.log_joint(sp_output_times, sp_target)
            ## log_total = log posterior
            log_total = log_joint + log_eq


        ### Back Propagation with TensorFlow ###
        grad_parameters = tape.gradient(log_total, param)   # list of length num_neural_networks

        if(flag):
            ## if flag=True derive also wrt betas_trainable
            grad_log_betas = tape.gradient(log_total, betas_trainable)
        else:
            grad_log_betas = []
        ### for every i in 1,...,num_neural_networks:
        ### grad_parameters[i] is a list of 2*L elements
        ### e.g.  (matrixW_first_layer, bias_Vector_first_layer, matrixW_second_layer, bias_Vector_second_layer, ...
        ###        matrixW_third_layer, bias_Vector_third_layer, matrixW_fourth_layer, bias_Vector_fourth_layer, ...)

        ## delete tape
        del tape
        return grad_parameters, grad_log_betas, loss_1, loss_2, loss_d, log_total, log_likelihood, log_prior_w, log_eq

    def get_trainable_weights_flatten(grad_parameters):
        """!
        For SVGD alg.
        flatten the list of list of parameters in input into a tensor of shape (num_neural_networks, lenght_of_theta)
        """
        w = []
        ## for loop over num_neural_networks
        for i in range( len(grad_parameters) ):
            w_i = []
            ## for loop on list of parameter
            for param in grad_parameters[i] :
                ## reshape
                w_i.append(tf.reshape(param,[-1]))
            w.append(tf.concat(w_i, axis=0))

        ## return a tensor of shape=(num_neural_networks, num_total_parameters_theta)
        return tf.convert_to_tensor(w)

    def _from_vector_to_parameter(self,grad_theta):
        """!
        Transform the flatten vector of grad_theta back to the list form using
        the bayes_nn.architecture_nn
        Return a list of length num_neural_networks, where each item is a
        list that contains all the matrix weights and bias vector for each layer

        @param grad_theta tensor of shape=(num_neural_networks, num_total_parameters_theta)
        """

        par_grad_theta = [] # list to store all the below lists (one for each neural network)

        for i in range(self.num_neural_networks):   # loop over all the neural networks

            grad_theta_i = []   # list to store all components of a single neural network
            i_1 = 0
            # loop over architecture_nn to collect 3 numbers for each layer: l1,l2,l3
            # that are: l1,l2 -> dimension of weight matrix W, l3 dimension of bias vector (for each layer)
            for (l1,l2,l3) in self.bayes_nn.architecture_nn:

                i_2 = i_1 + l1*l2   # compute the right index (after counting for the weight matrix)
                i_3 = i_2 + l3  # compute the right index (after counting for the bias vector)

                # reconstruct the weights matrix using the dimension l1,l2 and using reshape method
                weights_layer = grad_theta[i, i_1:i_2].reshape((l1,l2)) # (W, weights matrix)

                # reconstruct the bias vector using the dimension l3 and using reshape method
                bias_layer = grad_theta[i, i_2:i_3].reshape((l3,))  # (b, bias vector)

                # append the matrix W for this layer
                grad_theta_i.append(weights_layer)

                # append the vector b for this layer
                grad_theta_i.append(bias_layer)

                i_1 = i_3   # update the first index

            # append the list grad_theta_i to the list par_grad_theta (for each i=1,...,num_neural_networks)
            par_grad_theta.append(grad_theta_i)

        return par_grad_theta

    # Train of the SVGD BPINN #
    def train_all(self, verbosity):
        """ Train using SVGD algorithm """

        rec_log_betaD = []  # list that collects all the log_betaD during training
        rec_log_betaR = []  # list that collects all the log_betaR during training
        LOSS = []   # list that collects total loss during training
        LOSS1 = []  # list that collects loss of pde during training
        LOSS2 = []  # list that collects loss of high gradients during training
        LOSSD = []  # list that collects loss of exact noisy data during training

        # get noisy sparse exact data
        sp_inputs ,sp_at,sp_v = self.datasets_class.get_exact_data_with_noise()
        sp_target = sp_at

        # training for epoch in self.epochs
        for epoch in tqdm(range(self.epochs), desc="SVGD", leave=False):
            # to compute the time for a single epoch
            epochtime = time.time()

            # tp store the losses
            loss_1_tot = 0.
            loss_2_tot = 0.
            loss_d_tot = 0.

            # loop over self.train_loader: minibatch training
            # batch_idx = 0,...,len(self.train_loader)-1
            # inputs -> batch of collocation points -> shape=(batch_size, n_input)
            for batch_idx,inputs in enumerate(self.train_loader):

                # use the method self.compute_backprop_gradients() to compute the gradients of theta (and losses)
                grad_parameters, grad_log_betas, loss_1, loss_2, loss_d, \
                    log_total, log_likelihood, log_prior_w, log_eq = \
                    self._compute_backprop_gradients(sp_inputs, sp_target, inputs)

                debug_print_flag = False
                if(batch_idx==(len(self.train_loader)-1) and debug_print_flag):
                    print("-------------------------------")
                    print("\n**********START DEBUG*************")
                    fin_epochtime = time.time()-epochtime
                    print("Log total:      " ,log_total.numpy()[0])
                    print("Log likelihood: ", log_likelihood.numpy()[0])
                    print("Log prior w:    ", log_prior_w.numpy()[0])
                    print("Log equation:   ", log_eq.numpy()[0])
                    print("time for this iteration = ", fin_epochtime)
                    print("***********END DEBUG**************")


                ## collect all the parts of log_posterior (log_likelihood, log_prior_w and log_eq)
                self.bayes_nn.data_logloss.append(log_likelihood)
                self.bayes_nn.prior_logloss.append(log_prior_w)
                self.bayes_nn.eikonal_logloss.append(log_eq)

                # reshape theta and grad_theta from lists to tensors, in order to compute Kxx and dxKxx

                ## build a flatten vector of theta
                theta = self.bayes_nn.get_trainable_weights_flatten() # shape=(num_neural_networks, num_total_parameters_theta)
                ## build a flatten vector of grad_theta
                grad_theta = self.get_trainable_weights_flatten(grad_parameters) # shape=(num_neural_networks, num_total_parameters_theta)

                ######## SVGD modify the gradients of theta #########
                # calculating the kernel matrix and its gradients
                Kxx, dxKxx = self._Kxx_dxKxx(theta)
                grad_logp = tf.linalg.matmul(Kxx, grad_theta)
                ### the minus sign is needed for apply_gradients ###
                grad_theta = - (grad_logp + dxKxx) / self.num_neural_networks # shape=(num_neural_networks, num_total_parameters_theta)

                # reshape back grad_theta from shape=(num_neural_networks, num_total_parameters_theta)
                # to (num_neural_networks, *list* )
                grad_theta_list = self._from_vector_to_parameter(grad_theta.numpy())
                theta = self.bayes_nn.get_trainable_weights()

                # optimizer step: apply_gradients of grad_theta_list to theta
                for i in range(self.num_neural_networks):
                    # optimizer[0] is the Adam optimizer for theta (learning_rate = self.lr)
                    self.optimizers[0].apply_gradients(zip(grad_theta_list[i], theta[i]))

                # if log betas are trainable do the same passage for theta also to log_betas
                if(self.bayes_nn.log_betas.betas_trainable_flag()):
                    log_betas = tf.convert_to_tensor(self.bayes_nn.log_betas.get_trainable_log_betas())
                    log_betas = tf.transpose(log_betas) #(5,2)
                    grad_log_betas = tf.convert_to_tensor(grad_log_betas)
                    grad_log_betas = tf.transpose(grad_log_betas) #(5,2)

                    Kxx, dxKxx = self._Kxx_dxKxx(log_betas)
                    grad_logp = tf.linalg.matmul(Kxx, grad_log_betas)
                    grad_log_betas = - (grad_logp + dxKxx) / self.num_neural_networks

                    grad_log_betas_list = []

                    for i in range(grad_log_betas.shape[1]):
                        grad_log_betas_list.append(grad_log_betas[:,i])

                    # optimizer[1] is the Adam optimizer for log_beta (learning_rate = self.lr_noise)
                    self.optimizers[1].apply_gradients(zip(grad_log_betas_list, self.bayes_nn.log_betas.get_trainable_log_betas()))

                # in the last batch collect the log_betas
                if(batch_idx == len(self.train_loader)-1):
                    rec_log_betaD.append(self.bayes_nn.log_betas.log_betaD.numpy())
                    rec_log_betaR.append(self.bayes_nn.log_betas.log_betaR.numpy())

                loss_1_tot += loss_1
                loss_2_tot += loss_2
                loss_d_tot += loss_d

                if (batch_idx % max(int(len(self.train_loader)/5),1) == 0):
                    loss = loss_1+loss_2
                    print('Train Epoch: ',
                        epoch,"   " ,100. * batch_idx / len(self.train_loader),"%   ", "Loss: ", loss.numpy(), "   Loss d: ", loss_d.numpy())

            # append the losses
            loss_pde_tot = loss_1_tot+loss_2_tot
            loss_pde_tot = loss_pde_tot.numpy() + loss_d_tot.numpy()
            LOSS.append(loss_pde_tot/len(self.train_loader))
            LOSS1.append(loss_1_tot.numpy()/len(self.train_loader))
            LOSS2.append(loss_2_tot.numpy()/len(self.train_loader))
            LOSSD.append(loss_d_tot.numpy()/len(self.train_loader))


        return rec_log_betaD, rec_log_betaR, LOSS,LOSS1,LOSS2,LOSSD
