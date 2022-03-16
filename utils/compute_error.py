import tensorflow as tf
import numpy as np
import os

class compute_error:
    """
    Useful class to compute errors and uncertainty quantification
    """

    def __init__(self, n_out_sol, n_out_par, bayes_nn, datasets_class, path_result):
        """!
        Constructor

        @param n_out_sol
        @param bayes_nn the Bayesian NN we have already trained
        @param dataset_class an object of dataset_class
        @param path_result where we are goint to save all the error/UQ results
        """

        self.n_out_sol = n_out_sol
        self.n_out_par = n_out_par
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
        ## get the domain data
        inputs, u_true, f_true = self.datasets_class.get_dom_data()
        ## we are going to use the numpy vector notations shape=(len, )
        u_true = u_true.squeeze() ## TO CHECK IN HIGHER DIMENSION
        f_true = f_true.squeeze() ## TO CHECK IN HIGHER DIMENSION

        ## compute our prediction mean and standard deviation
        u_NN, f_NN, u_std, f_std = self.bayes_nn.mean_and_std(inputs)

        ## collect the vectors (discard all the other dimensions)
        u_NN = u_NN[:,0]
        u_std = u_std[:,0]
        f_NN = f_NN[:,0]
        f_std = f_std[:,0]

        ## to compute the Mean Square Error
        loss_f = tf.keras.losses

        ## accruacy of at -> MSE between at_NN and at_true
        error_u = loss_f.MSE(u_NN, u_true).numpy()
        ## accuracy of v -> MSE between v_NN and v_true
        error_f = loss_f.MSE(f_NN,f_true).numpy()
        ## (compute the relative error)
        ## compute the norm of true vectors
        u_truenorm = loss_f.MSE(u_true ,tf.zeros_like(u_true)).numpy()
        f_truenorm = loss_f.MSE(f_true, tf.zeros_like(f_true)).numpy()

        ## compute the relative error
        relative_error_u = np.sqrt(error_u/u_truenorm)
        relative_error_f = np.sqrt(error_f/f_truenorm)

        print('relative error |u - u_true|/|u_true| =', relative_error_u)
        print('relative error |f - f_true|/|f_true| =', relative_error_f)

        ##################### UQ ####################
        ## UQ mean -> is the mean of standard deviations in the domain
        ## UQ max -> is the max of standard deviations in the domain

        ## Std u,f mean
        uq_u_mean = u_std.mean()
        uq_f_mean = f_std.mean()

        ## Std u,f max
        uq_u_max = u_std.max()
        uq_f_max = f_std.max()

        print('mean uq u is',uq_u_mean)
        print('mean uq f is',uq_f_mean)

        print('max  uq u is',uq_u_max)
        print('max  uq f is',uq_f_max)


        string_error = "relative error norm activation_times = " + str(relative_error_u) + "\n" + \
                            "relative error norm velocity = " + str(relative_error_f) + "\n" + \
                            "UQ activation_times mean = " + str(uq_u_mean) + "\n" + \
                            "UQ velocity mean = " + str(uq_f_mean) + "\n" + \
                            "UQ activation_times max = " + str(uq_u_max) + "\n" + \
                            "UQ velocity max = " + str(uq_f_max)

        ## save all the errors and UQs in UQ.txt
        path = os.path.join(self.path_result,"UQ.txt")
        with open(path, 'w') as f:
            f.write(string_error)

        errors = {"error_u":error_u,
                        "error_f":error_f,
                        "relative_error_u":relative_error_u,
                        "relative_error_f":relative_error_f,
                        "uq_u_mean":uq_u_mean,
                        "uq_f_mean":uq_f_mean,
                        "uq_u_max":uq_u_max,
                        "uq_f_max":uq_f_max
                    }

        return u_NN, f_NN, u_std, f_std, errors
