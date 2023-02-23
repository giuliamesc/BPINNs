import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm

class HMC(Algorithm):
    """
    Class for HMC training
    """
    def __init__(self, bayes_nn, param_method, debug_flag):
        super().__init__(bayes_nn, param_method, debug_flag)
        """
        burn_in : number of epochs to discard (at the beginning); raises exception and sets it to zero if it is bigger than number of epochs
        skip    : number of epochs to skip during sampling
        HMC_L   : number of leap-frog steps to perform
        HMC_dt  : time interval for thetas update
        """
        if param_method["burn_in"] >= param_method["epochs"]:
            raise Exception("Burn-in is too high for this number of epochs!")
        
        self.burn_in = param_method["burn_in"]
        self.skip    = param_method["HMC_skip"]
        self.HMC_L   = param_method["HMC_L"]
        self.HMC_dt  = param_method["HMC_dt"] 
        self.HMC_eta = param_method["HMC_eta"]
        self.selected = list()

    def __leapfrog_step(self, old_theta, r, dt): # Si potrebbe cancellare old_theta
        """ Performs one leap-frog step starting from previous values of theta/sigma and r/s """
        grad_theta = self.model.grad_loss(self.data_batch, self.__full_loss)
        r = r - grad_theta * dt / 2
        self.model.nn_params = old_theta + r * dt 
        grad_theta = self.model.grad_loss(self.data_batch, self.__full_loss)
        r = r - grad_theta * dt / 2
        return self.model.nn_params, r

    def __compute_alpha(self, h0, h1):
        """ Computation of acceptance probabilities alpha and sampling of p (logarithm of both quantities) """
        p     = np.log(np.random.uniform())
        alpha = min(0, -h1+h0) 
        if np.isnan(h1): alpha = float("-inf")
        return alpha, p

    def __accept_reject(self, theta, r):
        """ 
        Acceptance-Rejection step:
        - Evaluation of current and previous value of the Hamiltonian function
        - Update or repetition of thetas depending on the acceptance-rejection step
        - Computation of the overall acceptance rate
        """
        h0 = self.__hamiltonian(theta[0], r[0])
        h1 = self.__hamiltonian(theta[1], r[1])
        
        alpha, p = self.__compute_alpha(h0, h1)
        accept = alpha >= p
        
        if self.debug_flag:
            print(f"\th0: {h0 :1.3e}")
            print(f"\th1: {h1 :1.3e}")
            print(f"\th: {h0-h1 :1.3e}")
            print(f"\ta: {np.exp(alpha)*100 :1.2f}%")

        if accept:
            if self.debug_flag: print("\tACCEPT")
            theta = theta[1]
            self.selected.append(True)
        else:
            if self.debug_flag: print("\tREJECT")
            theta = theta[0]
            self.selected.append(False)

        acc_rate = f"{100*sum(self.selected)/len(self.selected):1.2f}%"
        if self.debug_flag:
            print(f"\tAR: {acc_rate}")
        if not self.debug_flag:
            self.epochs_loop.set_postfix({"Acc.Rate": acc_rate})

        return theta

    def __hamiltonian(self, theta, r):
        """ Evaluation of the Hamiltonian function """
        self.model.nn_params = theta
        u = self.model.loss_total(self.data_batch, self.__full_loss).numpy()
        v_r = r.ssum() * self.HMC_eta**2/2
        return u + v_r
    
    def sample_theta(self, theta_0):
        """ Samples one parameter vector given its previous value """
        self.__full_loss = self.curr_ep > self.burn_in
        r_0 = theta_0.normal(self.HMC_eta) 
        r   = r_0.copy()
        theta = theta_0.copy()
        for _ in range(self.HMC_L):
            theta, r = self.__leapfrog_step(theta, r, self.HMC_dt)
        return self.__accept_reject((theta_0,theta), (r_0,r))

    def select_thetas(self, thetas_train):
        """ Compute burn-in and samples with skip """
        self.selected = self.selected[self.burn_in::self.skip]
        thetas_train = thetas_train[self.burn_in::self.skip]
        return thetas_train

    def train_log(self):
        """ Report log of the training"""
        print('End training:')
        self.compute_time()
        accepted = sum(self.selected)
        rejected = len(self.selected) - accepted
        print(f"\tAccepted Values: {accepted}/{accepted+rejected} {100*accepted/(accepted+rejected) :1.2f}%")
