import tensorflow as tf
from algorithms.Algorithm import Algorithm

class ADAM(Algorithm):
    """
    Class for ADAM (deterministic) training
    """
    def __init__(self, bayes_nn, param_method, debug_flag):
        super().__init__(bayes_nn, param_method, debug_flag)
        self.burn_in = param_method.get("burn_in", self.epochs)
        self.beta_1  = param_method["beta_1"]
        self.beta_2  = param_method["beta_2"]
        self.eps     = param_method["eps"]
        self.lr      = param_method["lr"]
        self.__initialize_momentum()

    def __initialize_momentum(self):
        self.num_2layers = len(self.model.nn_params)
        self.m = [0.0 for _ in range(self.num_2layers)]
        self.v = [0.0 for _ in range(self.num_2layers)]

    def sample_theta(self, theta_0):
        """ Samples one parameter vector given its previous value """
        full_loss = self.curr_ep > self.burn_in
        grad_theta = self.model.grad_loss(self.data_batch, full_loss)
        theta = theta_0.copy()
        for i in range(self.num_2layers):
            self.m[i] = self.beta_1*self.m[i] + (1-self.beta_1)*grad_theta[i]
            self.v[i] = self.beta_2*self.v[i] + (1-self.beta_2)*(grad_theta[i]*grad_theta[i])
            theta[i] -= self.lr*(self.m[i]/(1-self.beta_1))/(tf.math.sqrt(self.v[i]/(1-self.beta_2))+self.eps)
        return theta

    def select_thetas(self, thetas_train):
        """ Compute burn-in and skip samples """
        return thetas_train[-1:]