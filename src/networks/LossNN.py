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
        # Choice of loss to be used
        self.metric = [k for k,v in par.metrics.items() if v]
        self.keys   = [k for k,v in  par.losses.items() if v]
        self.vars   = par.uncertainty 

    @staticmethod
    def __mse_theta(theta, dim):
        """ Sum of Squared Errors """
        return sum([tf.norm(t)**2 for t in theta]) / dim

    @staticmethod
    def __mse(vect):
        """ Mean Squared Error """
        norm = tf.norm(vect, axis = -1)
        return tf.keras.losses.MSE(norm, tf.zeros_like(norm))

    @staticmethod
    def __normal_loglikelihood(mse, n, log_var):
        """ Negative log-likelihood """
        return 0.5 * n * ( mse * tf.math.exp(log_var) - log_var) # delete * n in the laplace case?

    def __loss_data(self, outputs, targets, log_var):
        """ Auxiliary loss function for the computation of Normal(output | target, 1 / beta * I) """
        post_data = self.__mse(outputs-targets)
        log_data = self.__normal_loglikelihood(post_data, outputs.shape[0], log_var)
        return self.tf_convert(post_data), self.tf_convert(log_data)

    def __loss_data_u(self, data):
        """ Fitting loss on u; computation of the residual at points of measurement of u """
        outputs = self.forward(data["dom"])
        log_var = tf.math.log(1/self.vars["sol"]**2)
        return self.__loss_data(outputs[0], data["sol"], log_var)

    def __loss_data_f(self, data):
        """ Fitting loss on f; computation of the residual at points of measurement of f """
        outputs = self.forward(data["dom"])
        log_var = tf.math.log(1/self.vars["par"]**2)
        return self.__loss_data(outputs[1], data["par"], log_var)

    def __loss_data_b(self, data):
        """ Boundary loss; computation of the residual on boundary conditions """
        outputs = self.forward(data["dom"])
        log_var = tf.math.log(1/self.vars["bnd"]**2)
        return self.__loss_data(outputs[0], data["sol"], log_var)

    def __loss_residual(self, data):
        """ Physical loss; computation of the residual of the PDE """
        inputs = self.tf_convert(data["dom"])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            u, f = self.forward(inputs)
            residuals = self.pinn.comp_residual(inputs, u, f, tape)
        mse = self.__mse(residuals)
        log_var =  tf.math.log(1/self.vars["pde"]**2)
        log_res = self.__normal_loglikelihood(mse, inputs.shape[0], log_var)
        return mse, log_res

    def __loss_prior(self):
        """ Prior for neural network parameters, assuming them to be distributed as a gaussian N(0,stddev^2) """
        log_var = tf.math.log(1/self.stddev**2)
        prior   = self.__mse_theta(self.model.trainable_variables, self.dim_theta)
        loglike = self.__normal_loglikelihood(prior, self.dim_theta, log_var)
        return prior, loglike

    def __compute_loss(self, dataset, keys, full_loss = True):
        """ Computation of the losses listed keys """
        pst, llk = dict(), dict()
        if "data_u" in keys: pst["data_u"], llk["data_u"] = self.__loss_data_u(dataset.data_sol)
        if "data_f" in keys: pst["data_f"], llk["data_f"] = self.__loss_data_f(dataset.data_par)
        if "data_b" in keys: pst["data_b"], llk["data_b"] = self.__loss_data_b(dataset.data_bnd)
        if "prior"  in keys: pst["prior"],  llk["prior"]  = self.__loss_prior()
        if "pde"    in keys: pst["pde"], llk["pde"] = self.__loss_residual(dataset.data_pde) if full_loss else (0.0, 0.0)
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
        grad_thetas = tape.gradient(diff_llk, self.model.trainable_variables)

        return grad_thetas

