import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm

class HMC(Algorithm):
    """
    Class for HMC training
    """
    def __init__(self, bayes_nn, param_method, debug_flag):
        super().__init__(bayes_nn, param_method, debug_flag)
        
        if param_method["burn_in"] >= param_method["epochs"]:
            raise Exception("Burn-in is too high for this number of epochs!")
        self.burn_in = param_method["burn_in"]
        self.HMC_L   = param_method["HMC_L"]
        self.HMC_dt  = param_method["HMC_dt"]
        self.eta = 0.01
        self.selected = list()

    def __leapfrog_step(self, old_theta, r, dt): # SI potrebbe cancellare old_theta
        """ COMMENTARE """
        r = [ x - y * dt/2 for x,y in zip(r, self.model.grad_loss(self.data))]
        self.model.nn_params = [ x + y * dt for x,y in zip(old_theta, r)]
        r = [ x - y * dt/2 for x,y in zip(r, self.model.grad_loss(self.data))]
        return self.model.nn_params, r

    def __compute_alpha(self, h0, h1):
        p     = np.log(np.random.uniform())
        alpha = min(0, -h1+h0)  #alpha = min(0, +h1-h0) # Alpha value 
        if np.isnan(h1): alpha = float("-inf") # Avoid NaN values
        return alpha, p

    def __accept_reject(self, theta_0, theta_1, r_0, r_1):
        """ COMMENTARE E DEBUGGARE """
        h0 = self.__hamiltonian(theta_0, r_0)
        h1 = self.__hamiltonian(theta_1, r_1)
        
        alpha, p = self.__compute_alpha(h0, h1)
        accept = alpha >= p #accept = alpha <= p ??
        
        if self.debug_flag:
            print(f"\th0: {h0 :1.3e}")
            print(f"\th1: {h1 :1.3e}")
            print(f"\th: {h0-h1 :1.3e}") #print(f"\t h: {h1-h0 :1.3e}") ??
            print(f"\ta: {np.exp(alpha)*100 :1.2f}%")

        if accept:
            if self.debug_flag: print("\tACCEPT")
            theta = theta_1
            self.selected.append(True)
        else:
            if self.debug_flag: print("\tREJECT")
            theta = theta_0
            self.selected.append(False)
        
        return theta 

    def __hamiltonian(self, theta, r):
        """ COMMENTARE """
        self.model.nn_params = theta
        u = self.model.loss_total(self.data)[1]["Total"].numpy()
        v = sum([tf.norm(t).numpy()**2 for t in r]) * self.eta**2/2
        return u + v
    
    def sample_theta(self, theta_0):
        """ COMMENTARE """
        r_0 = [tf.random.normal(x.shape) for x in theta_0]
        r     = r_0.copy()
        theta = theta_0.copy()
        for _ in range(self.HMC_L):
            theta, r = self.__leapfrog_step(theta, r, self.HMC_dt)
        return self.__accept_reject(theta_0, theta, r_0, r)

    def select_thetas(self, thetas_train):
        """ Compute burn-in and skip samples """
        self.selected = self.selected[self.burn_in:]
        return thetas_train[self.burn_in:]

    def train_log(self):
        """ Report log of the training"""
        print('End training:')
        self.compute_time()
        accepted = sum(self.selected)
        rejected = len(self.selected) - accepted
        print(f"\tAccepted Values: {accepted}/{accepted+rejected} {100*accepted/(accepted+rejected) :1.2f}%")
