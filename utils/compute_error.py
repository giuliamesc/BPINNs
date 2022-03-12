import tensorflow as tf
import numpy as np
import os

class compute_error:
    """
    Useful class to compute errors and uncertainty quantification
    """

    def __init__(self, n_output_vel, bayes_nn, datasets_class, path_result):
        """!
        Constructor

        @param n_output_vel
        @param bayes_nn the Bayesian NN we have already trained
        @param dataset_class an object of dataset_class
        @param path_result where we are goint to save all the error/UQ results
        """

        ## n_output_vel
        self.n_output_vel = n_output_vel
        ## Bayesian neural network
        self.bayes_nn = bayes_nn
        ## Datasets
        self.datasets_class = datasets_class
        ## path for the results
        self.path_result = path_result

    def error(self):
        """
        Compute errors and UQs
        """
        ## distinguish between Isotropic, (v_scalar) and Anisotropic case

        if self.n_output_vel ==1: # Isotropic case
            ## get the domain data
            inputs,at_true,v_true = self.datasets_class.get_dom_data()
            ## we are going to use the numpy vector notations shape=(len, )
            at_true = at_true.squeeze()
            v_true = v_true.squeeze()

            ## compute our prediction mean and standard deviation
            ## (at_NN) -> at mean
            ## (at_std) -> at standard deviation (same for v)
            at_NN, v_NN, at_std, v_std = self.bayes_nn.mean_and_std(inputs)

            ## collect the vectors (discard all the other dimensions)
            at_NN = at_NN[:,0]
            at_std = at_std[:,0]
            v_NN = v_NN[:,0,0]
            v_std = v_std[:,0,0]

            ## to compute the Mean Square Error
            loss_f = tf.keras.losses

            ## accruacy of at -> MSE between at_NN and at_true
            error_at = loss_f.MSE(at_NN, at_true).numpy()

            ## accuracy of v -> MSE between v_NN and v_true
            error_v = loss_f.MSE(v_NN,v_true).numpy()

            ## (compute the relative error)
            ## compute the norm of true vectors
            at_truenorm = loss_f.MSE(at_true ,tf.zeros_like(at_true)).numpy()
            v_truenorm = loss_f.MSE(v_true, tf.zeros_like(v_true)).numpy()

            ## compute the relative error
            relative_error_at = np.sqrt(error_at/at_truenorm)
            relative_error_v = np.sqrt(error_v/v_truenorm)

            print('relative error |u - u_true|/|u_true| =', relative_error_at)
            print('relative error |f - f_true|/|f_true| =', relative_error_v)

            ##################### UQ ####################
            ## UQ mean -> is the mean of standard deviations in the domain
            ## UQ max -> is the max of standard deviations in the domain

            ## Std at mean
            uq_at_mean = at_std.mean()
            ## Std v mean
            uq_v_mean = v_std.mean()

            ## Std at max
            uq_at_max = at_std.max()
            ## Std v max
            uq_v_max = v_std.max()

            print('mean uq u is',uq_at_mean)
            print('mean uq f is',uq_v_mean)

            print('max uq u is',uq_at_max)
            print('max uq f is',uq_v_max)


            string_error = "relative error norm activation_times = " + str(relative_error_at) + "\n" + \
                            "relative error norm velocity = " + str(relative_error_v) + "\n" + \
                            "UQ activation_times mean = " + str(uq_at_mean) + "\n" + \
                            "UQ velocity mean = " + str(uq_v_mean) + "\n" + \
                            "UQ activation_times max = " + str(uq_at_max) + "\n" + \
                            "UQ velocity max = " + str(uq_v_max)

            ## save all the errors and UQs in UQ.txt
            path = os.path.join(self.path_result,"UQ.txt")
            with open(path, 'w') as f:
                f.write(string_error)

            errors = {"error_at":error_at,
                        "error_v":error_v,
                        "relative_error_at":relative_error_at,
                        "relative_error_v":relative_error_v,
                        "uq_at_mean":uq_at_mean,
                        "uq_v_mean":uq_v_mean,
                        "uq_at_max":uq_at_max,
                        "uq_v_max":uq_v_max
                    }


        else: #Anisotropic
            # get domain data
            inputs,at_true,v_true = self.datasets_class.get_dom_data()
            at_true = at_true.squeeze()

            # compute mean and std
            at_NN, v_NN, at_std, v_std = self.bayes_nn.mean_and_std(inputs)

            # collect vector of at
            at_NN = at_NN[:,0]
            at_std = at_std[:,0]

            # collect tensors of v = [a,b,c]
            v_NN = v_NN[:,0,:]
            v_std = v_std[:,0,:]

            ### to compute the MSError
            loss_f = tf.keras.losses

            # accruacy of at -> MSE between at_NN and at_true
            error_at = loss_f.MSE(at_NN,at_true).numpy()

            # accuracy of v -> MSE between v_NN and v_true for each entries
            error_v = []
            for i in range(v_NN.shape[1]):
                error_v.append(loss_f.MSE(v_NN[:,i],v_true[:,i]).numpy())
            #print("error_at: ", error_at)

            ## compute the relative error
            at_truenorm = loss_f.MSE(at_true ,tf.zeros_like(at_true)).numpy()

            v_truenorm = []
            for i in range(v_NN.shape[1]):
                v_truenorm.append(loss_f.MSE(v_true[:,i], tf.zeros_like(v_true[:,i])).numpy())

            relative_error_at = np.sqrt(error_at/at_truenorm)
            relative_error_v = []
            for i in range(v_NN.shape[1]):
                relative_error_v.append(np.sqrt(error_v[i]/v_truenorm[i]))

            print('relative error |u - u_true|/|u_true| =', relative_error_at)
            print('relative error |f - f_true|/|f_true| =', relative_error_v)

            ##################### UQ ####################
            ## Std at mean
            uq_at_mean = at_std.mean()
            ## Std v mean
            uq_v_mean = v_std.mean(axis=0)

            ## Std at max
            uq_at_max = at_std.max()
            ## Std v max
            uq_v_max = v_std.max(axis=0)

            print('mean uq u is',uq_at_mean)
            print('mean uq f is',uq_v_mean)

            #
            print('max uq u is',uq_at_max)
            print('max uq f is',uq_v_max)


            string_error = "relative error norm activation_times = " + str(relative_error_at) + "\n" + \
                            "relative error norm velocity = " + str(relative_error_v) + "\n" + \
                            "UQ activation_times mean = " + str(uq_at_mean) + "\n" + \
                            "UQ velocity mean = " + str(uq_v_mean) + "\n" + \
                            "UQ activation_times max = " + str(uq_at_max) + "\n" + \
                            "UQ velocity max = " + str(uq_v_max)


            path = os.path.join(self.path_result,"UQ.txt")
            with open(path, 'w') as f:
                f.write(string_error)

            errors = {"error_at":error_at,
                        "error_v":error_v,
                        "relative_error_at":relative_error_at,
                        "relative_error_v":relative_error_v,
                        "uq_at_mean":uq_at_mean,
                        "uq_v_mean":uq_v_mean,
                        "uq_at_max":uq_at_max,
                        "uq_v_max":uq_v_max
                    }

        return at_NN, v_NN, at_std, v_std, errors
