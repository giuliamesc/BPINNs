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
        self.m = self.model.nn_params*0
        self.v = self.model.nn_params*0

    def sample_theta(self, theta_0):
        """ Samples one parameter vector given its previous value """
        full_loss  = self.curr_ep > self.burn_in
        grad_theta = self.model.grad_loss(self.data_batch, full_loss)
        theta = theta_0.copy()
        
        self.m = self.m*self.beta_1 + (grad_theta)*(1-self.beta_1)
        self.v = self.v*self.beta_2 + (grad_theta**2)*(1-self.beta_2)
        theta -= (self.m/(1-self.beta_1)*self.lr) / ((self.v/(1-self.beta_2))**0.5+self.eps)

        return theta

    def select_thetas(self, thetas_train):
        """ Compute burn-in and skip samples """
        return thetas_train[-1:]