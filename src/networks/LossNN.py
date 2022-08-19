from .CoreNN import CoreNN
import tensorflow as tf

class LossNN(CoreNN):
    """
    - Evaluate PDEs residuals (using pde constraint)
    - Compute mean-squared-errors and loglikelihood
        - residual loss (pdes)
        - data loss (fitting)
        - prior loss WIP

    Losses structure
    - loss_total: tuple (mse, loglikelihood)
    - mse, loglikelihood: dictionaries with keys:
        - res  : evaluated in collocation pts with physical losses
        - data : evaluated in fitting pts with targets value
        - prior: WIP
        - Total: sum of the previous
    """

    def __init__(self, par, comp_res, **kw):

        super(LossNN, self).__init__(par, **kw)
        # Parameters for combining losses
        self.coeff  = par.coeff
        self.sigmas = par.sigmas
        # Function for residual evaluation
        self.compute_residual = comp_res
        
    def __loss_residual(self, inputs):
        """
        Computes the MSE and log-likelihood of the data 
        inputs: (num_collocation, n_input)
        """
        # compute loss using pde_constraint
        pde_res = self.compute_residual(inputs, self.forward)
        mse_res = self.__mse(pde_res)

        # log loss for a Gaussian -> Normal(loss_1 | zeros, 1/betaR*Identity)
        n_r = pde_res.shape[0] # number of samples
        log_var = self.sigmas["pde_pn"] # log(1/betaR)

        log_res = self.__normal_loglikelihood(mse_res, n_r, log_var)
        log_res *= self.coeff["res"]

        return self.__convert(mse_res), self.__convert(log_res)

    def __loss_data(self, inputs, targets):
        """
        Computes the MSE and log-likelihood of the data 
        inputs  : np(num_fitting, n_input)
        targets : np(num_fitting, n_out_sol)
        outputs : tf(num_fitting, n_out_sol)
        """
        # Normal(output | target, 1 / betaD * I)
        outputs, _ = self.forward(inputs, split = True)
        mse_data = self.__mse(outputs-targets)

        n_d = outputs.shape[0]
        log_var = self.sigmas["data_pn"] # log(1/betaD)

        log_data = self.__normal_loglikelihood(mse_data, n_d, log_var)
        log_data*= self.coeff["data"]

        return self.__convert(mse_data), self.__convert(log_data)

    def __loss_prior(self):
        """
        Compute the logloss of the prior 
        AGGIUNGI DIMENSIONI
        """
        loss_prior = 0.
        log_prior = 0.
        # compute log prior of w (t-student)
        log_prior *= self.coeff["prior"]
        return loss_prior, log_prior

    @staticmethod
    def __convert(tensor): 
        """ Conversion of a numpy array to tensor """
        return tf.cast(tensor, dtype=tf.float32)

    @staticmethod
    def __mse(vect):
        """ Mean Squared Error """
        norm = tf.norm(vect, axis = -1)
        return tf.keras.losses.MSE(norm, tf.zeros_like(norm))

    @staticmethod
    def __normal_loglikelihood(mse, n, log_var):
        """ Negative log-likelihood """
        """ It's a consistent estimator?? """
        return -1*(- 0.5 * n * mse * tf.math.exp(log_var) + 0.5 * n * log_var)

    def loss_total(self, dataset):
        """ Creation of the dictionary containing all MSEs and log-likelihoods """
        loss, logloss = dict(), dict()
        loss["res"],   logloss["res"]   = self.__loss_residual(dataset.coll_data[0])
        loss["data"],  logloss["data"]  = self.__loss_data(dataset.exact_data[0], dataset.exact_data[1])
        loss["prior"], logloss["prior"] = self.__loss_prior()
        loss["Total"], logloss["Total"] = sum(loss.values()), sum(logloss.values())
        return loss, logloss

