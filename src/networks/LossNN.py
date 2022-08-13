from .CoreNN import CoreNN
import tensorflow as tf

class LossNN(CoreNN):
    
    """
    ***** Key Features *****
    - Evaluate PDEs residuals (using pde constraint)
    - Compute losses and loglosses
        - residual loss (pdes)
        - data loss (fitting)
        - prior loss WIP
    
    **** Other Features ****
    - Likelihood evaluation
    - Tensor casting in float32
    """

    def __init__(self, comp_res, **kw):

        super(LossNN, self).__init__(**kw)
        self.compute_residual = comp_res

    def loss_total(self, dataset):
        loss, logloss = dict(), dict()
        loss["res"],   logloss["res"]   = self.__loss_residual(dataset.coll_data[0])
        loss["data"],  logloss["data"]  = self.__loss_data(dataset.exact_data_noise[0], dataset.exact_data_noise[1])
        loss["prior"], logloss["prior"] = self.__loss_prior()
        loss["Total"], logloss["Total"] = sum(loss.values()), sum(logloss.values())
        return loss, logloss

    @tf.function # decorator to speed up the computation
    def __loss_residual(self, inputs):
        """
        Compute the loss and logloss of the pde
        AGGIUNGI DIMENSIONI
        """
        # compute loss using pde_constraint
        pde_res = self.compute_residual(inputs, self.forward)
        mse_res = tf.reduce_mean(tf.keras.losses.MSE(pde_res, tf.zeros_like(pde_res)))

        # log loss for a Gaussian -> Normal(loss_1 | zeros, 1/betaR*Identity)
        n_r = pde_res.shape[0] # number of samples
        log_var = self.par.sigmas["pde_prior_noise"] # log(1/betaR)

        log_res = self.normal_loglikelihood(mse_res, n_r, log_var)
        log_res *= self.par.param_res

        return self.convert(mse_res), self.convert(log_res)

    @tf.function # decorator to speed up the computation
    def __loss_data(self, inputs, targets):
        """
        Compute the loss and logloss of the data 
        AGGIUNGI DIMENSIONI
        """
        # Normal(output | target, 1 / betaD * I)
        outputs, _ = self.forward(inputs, split = True)
        mse_data = tf.reduce_mean(tf.keras.losses.MSE(outputs, targets))

        n_d = outputs.shape[0]
        log_var = self.par.sigmas["data_prior_noise"] # log(1/betaD)

        log_data = self.normal_loglikelihood(mse_data, n_d, log_var)
        log_data*= self.par.param_data

        return self.convert(mse_data), self.convert(log_data)

    @tf.function # decorator to speed up the computation
    def __loss_prior(self):
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
