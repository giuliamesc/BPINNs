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
        self.keys = ("Total", "data_u") # Total is mandatory
        self.coeff  = par.coeff

    @staticmethod
    def __mse(vect):
        """ Mean Squared Error """
        norm = tf.norm(vect, axis = -1)
        return tf.keras.losses.MSE(norm, tf.zeros_like(norm))

    @staticmethod
    def __normal_loglikelihood(mse, n, log_var):
        """ Negative log-likelihood """
        return 0.5 * n * mse * tf.math.exp(log_var) - 0.5 * n * log_var

    def __loss_data(self, outputs, targets):
        # Normal(output | target, 1 / betaD * I)
        post_data = self.__mse(outputs-targets)
        log_var  = self.sg_params[0][0] # log(1/betaD)
        log_data = self.__normal_loglikelihood(post_data, outputs.shape[0], log_var)
        return self.tf_convert(post_data), self.tf_convert(log_data)

    def __loss_data_u(self, dataset):
        outputs = self.forward(dataset.noise_data[0])
        return self.__loss_data(outputs[0], dataset.noise_data[1])

    def __loss_data_f(self, dataset):
        outputs = self.forward(dataset.noise_data[0])
        return self.__loss_data(outputs[1], dataset.noise_data[2])

    def __loss_data_b(self):
        return 0.0, 0.0

    def __loss_prior(self):
        posterior_prior = 0.
        loglike_prior   = 0.
        return posterior_prior, loglike_prior

    def loss_total(self, dataset):
        """ Creation of the dictionary containing all posteriors and log-likelihoods """
        pst, llk = dict(), dict()
        if "data_u" in self.keys: pst["data_u"], llk["data_u"] = self.__loss_data_u(dataset)
        if "data_f" in self.keys: pst["data_f"], llk["data_f"] = self.__loss_data_f(dataset)
        if "prior"  in self.keys: pst["prior"],  llk["prior"]  = self.__loss_prior()
        pst["Total"] = sum(pst.values())
        llk["Total"] = sum(llk.values())
        return pst, llk

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

