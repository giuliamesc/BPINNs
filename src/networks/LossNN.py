from .PhysNN import PhysNN
import tensorflow as tf

class LossNN(PhysNN):
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

    def __init__(self, par, **kw):

        super(LossNN, self).__init__(par, **kw)
        self.coeff  = par.coeff
        self.sg_params = [self.__convert([par.sigmas["data_pn"], par.sigmas["pde_pn"]])]
        self.sg_flags  = [par.sigmas["data_pn_flag"], par.sigmas["pde_pn_flag"]]
        self.sigmas = list()

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
        return 0.5 * n * mse * tf.math.exp(log_var) - 0.5 * n * log_var

    def __loss_residual(self, _): return 0.0, 0.0 # DEPRECATED

    def __loss_data(self, inputs, targets):
        """
        Computes the MSE and log-likelihood of the data 
        inputs  : np(num_fitting, n_input)
        targets : np(num_fitting, n_out_sol)
        outputs : tf(num_fitting, n_out_sol)
        """
        # Normal(output | target, 1 / betaD * I)
        outputs, _  = self.forward(inputs)
        mse_data = self.__mse(outputs-targets)

        n_d = outputs.shape[0]
        log_var = self.sg_params[0][0] # log(1/betaD)

        log_data = self.__normal_loglikelihood(mse_data, n_d, log_var)
        log_data*= self.coeff["data"]

        return self.__convert(mse_data), self.__convert(log_data)

    def __loss_prior(self):
        posterior_prior = 0.
        loglike_prior   = 0.
        return posterior_prior, loglike_prior

    def loss_total(self, dataset):
        """ Creation of the dictionary containing all posteriors and log-likelihoods """
        posterior, loglike = dict(), dict()
        posterior["res"],   loglike["res"]   = self.__loss_residual(dataset.coll_data[0])
        posterior["data"],  loglike["data"]  = self.__loss_data(dataset.noise_data[0], dataset.noise_data[1])
        posterior["prior"], loglike["prior"] = self.__loss_prior()
        posterior["Total"], loglike["Total"] = sum(posterior.values()), sum(loglike.values())
        return posterior, loglike

    def grad_loss(self, dataset):

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            tape.watch(self.sg_params)
            _, loglike = self.loss_total(dataset)
        
        grad_thetas = tape.gradient(loglike["Total"], self.model.trainable_variables)
        grad_sigmas = tape.gradient(loglike["Total"], self.sg_params)
        
        if not self.sg_flags[0]: grad_sigmas[0] *= [0.0, 1.0] # if data prior noise not trainable
        if not self.sg_flags[1]: grad_sigmas[0] *= [1.0, 0.0] # if  pde prior noise not trainable
        
        return grad_thetas, grad_sigmas

