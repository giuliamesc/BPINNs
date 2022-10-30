from .PhysNN import PhysNN
import tensorflow as tf

class LossNN(PhysNN):
    """
    Evaluate PDEs residuals (using pde constraint)
    Compute mean-squared-errors and loglikelihood
        - residual loss (pdes)
        - boundary loss (boundary conditions)
        - data loss (fitting)
        - prior loss
    Losses structure
        - loss_total: tuple (mse, loglikelihood)
        - mse, loglikelihood: dictionaries with keys relative to loss type
    """

    def __init__(self, par, **kw):
        super(LossNN, self).__init__(par, **kw)
        self.sigmas = [par.sigmas["data_pn"]]
        self.metric = ["data_u", "data_f", "pde"] # Must be added in config ??
        self.keys   = ["data_u", "data_f", "pde"] # Must be added in config ??

    @staticmethod
    def __sse_theta(theta):
        """ Sum of Squared Errors """
        return sum([tf.norm(t)**2 for t in theta])

    @staticmethod
    def __mse(vect):
        """ Mean Squared Error """
        norm = tf.norm(vect, axis = -1)
        return tf.keras.losses.MSE(norm, tf.zeros_like(norm))

    @staticmethod
    def __normal_loglikelihood(mse, n, log_var):
        """ Negative log-likelihood """
        return 0.5 * n * ( mse * tf.math.exp(log_var) - log_var) # delete * n in the laplace case?

    def __loss_data(self, outputs, targets):
        """ Auxiliary loss function for the computation of Normal(output | target, 1 / beta * I) """
        post_data = self.__mse(outputs-targets)
        log_var  = self.sigmas[0]
        log_data = self.__normal_loglikelihood(post_data, outputs.shape[0], log_var)
        return self.tf_convert(post_data), self.tf_convert(log_data)

    def __loss_data_u(self, dataset):
        """ Fitting loss on u; computation of the residual at points of measurement of u """
        outputs = self.forward(dataset.noise_data[0])
        return self.__loss_data(outputs[0], dataset.noise_data[1])

    def __loss_data_f(self, dataset):
        """ Fitting loss on f; computation of the residual at points of measurement of f """
        outputs = self.forward(dataset.noise_data[0])
        return self.__loss_data(outputs[1], dataset.noise_data[2])

    def __loss_data_b(self):
        """ Boundary loss; computation of the residual on boundary conditions """
        return 0.0, 0.0

    def __loss_residual(self, dataset, coeff = 1e-3):
        """ Physical loss; computation of the residual of the PDE """
        inputs = self.tf_convert(dataset.coll_data[0])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            u, f = self.forward(inputs)
            residuals = self.pinn.comp_residual(inputs, u, f, tape)
        mse = coeff * self.__mse(residuals)
        log_var = self.sigmas[0]
        log_res = coeff * self.__normal_loglikelihood(mse, inputs.shape[0], log_var)
        return mse, log_res

    def __loss_prior(self):
        """ Prior for neural network parameters, assuming them to be distributed as a gaussian N(0,stddev^2) """
        log_var = tf.math.log(self.stddev**2)
        prior   = self.__sse_theta(self.nn_params) / self.dim_theta
        loglike = self.__normal_loglikelihood(prior, self.dim_theta, log_var) / self.dim_theta
        return prior, loglike

    def __compute_loss(self, dataset, keys, full_loss = True):
        """ Computation of the losses listed keys """
        pst, llk = dict(), dict()
        if "data_u" in keys: pst["data_u"], llk["data_u"] = self.__loss_data_u(dataset)
        if "data_f" in keys: pst["data_f"], llk["data_f"] = self.__loss_data_f(dataset)
        if "data_b" in keys: pst["data_b"], llk["data_b"] = self.__loss_data_b(dataset)
        if "prior"  in keys: pst["prior"],  llk["prior"]  = self.__loss_prior()
        if "pde"    in keys: pst["pde"], llk["pde"] = self.__loss_residual(dataset, coeff = 1e-3) if full_loss else (0.0, 0.0)
        return pst, llk

    def metric_total(self, dataset, full_loss = True):
        """ Computation of the losses required to be tracked """
        pst, llk = self.__compute_loss(dataset, self.metric, full_loss)
        pst["Total"] = sum(pst.values())
        llk["Total"] = sum(llk.values())
        return pst, llk

    def loss_total(self, dataset, full_loss = True):
        """ Creation of the dictionary containing all posteriors and log-likelihoods """
        _, llk = self.__compute_loss(dataset, self.keys, full_loss)
        return sum(llk.values())

    def grad_loss(self, dataset, full_loss = True):
        """ Computation of the gradient of the loss function with respect to the network trainable parameters """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            diff_llk = self.loss_total(dataset, full_loss)
        grad_thetas = tape.gradient(diff_llk, self.model.trainable_variables)  ## ADD GRAD LAMBDA

        return grad_thetas

