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
        self.skip    = param_method["skip"]
        self.HMC_L   = param_method["HMC_L"]
        self.HMC_dt  = param_method["HMC_dt"] 
        self.HMC_eta = param_method["HMC_eta"]
        self.selected = list()

    def __check_trainable(self, s):
        """ If the sigmas are not trainable, sets sigma vector to zero """
        if not self.model.sg_flags[0]: s[0] *= [0.0, 1.0]
        if not self.model.sg_flags[1]: s[0] *= [1.0, 0.0]
        return s

    def __leapfrog_step(self, old_theta, old_sigma, r, s, dt): # SI potrebbe cancellare old_theta
        """ Performs one leap-frog step starting from previous values of theta/sigma and r/s """
        ds = dt*1e-2

        grad_theta, grad_sigma = self.model.grad_loss(self.data)
        r = [ x - y * dt/2 for x,y in zip(r, grad_theta)]
        s = [ x - y * ds/2 for x,y in zip(s, grad_sigma)]

        self.model.nn_params = [ x + y * dt for x,y in zip(old_theta, r)]
        self.model.sg_params = [ x + y * ds for x,y in zip(old_sigma, s)] 

        grad_theta, grad_sigma = self.model.grad_loss(self.data)
        r = [ x - y * dt/2 for x,y in zip(r, grad_theta)]
        s = [ x - y * ds/2 for x,y in zip(s, grad_sigma)]
        
        return self.model.nn_params, self.model.sg_params, r, s

    def __compute_alpha(self, h0, h1):
        """ Computation of acceptance probabilities alpha and sampling of p (logarithm of both quantities) """
        p     = np.log(np.random.uniform())
        alpha = min(0, -h1+h0)  #alpha = min(0, +h1-h0) # Alpha value 
        if np.isnan(h1): alpha = float("-inf") # Avoid NaN values
        return alpha, p

    def __accept_reject(self, theta, sigma, r, s):
        """ 
        Acceptance-Rejection step:
        - Evaluation of current and previous value of the Hamiltonian function
        - Update or repetition of thetas depending on the acceptance-rejection step
        - Computation of the overall acceptance rate
        """
        h0 = self.__hamiltonian(theta[0], sigma[0], r[0], s[0])
        h1 = self.__hamiltonian(theta[1], sigma[1], r[1], s[1])
        
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
            sigma = sigma[1]
            self.selected.append(True)
        else:
            if self.debug_flag: print("\tREJECT")
            theta = theta[0]
            sigma = sigma[1]
            self.selected.append(False)

        acc_rate = f"{100*sum(self.selected)/len(self.selected):1.2f}%"
        if self.debug_flag:
            print(f"\tAR: {acc_rate}")
        if not self.debug_flag:
            self.epochs_loop.set_postfix({"Acc.Rate": acc_rate})
        
        return theta, sigma

    def __hamiltonian(self, theta, sigma, r, s):
        """ Evaluation of the Hamiltonian function """
        self.model.nn_params = theta
        self.model.sg_params = sigma
        u = self.model.loss_total(self.data)[1]["Total"].numpy()
        v_r = sum([tf.norm(t).numpy()**2 for t in r]) * self.HMC_eta**2/2
        v_s = sum([tf.norm(t).numpy()**2 for t in s]) * self.HMC_eta**2/2
        return u + v_r + v_s
    
    def sample_theta(self, theta_0, sigma_0):
        """ Samples one parameter vector given its previous value """
        r_0 = [tf.random.normal(x.shape, stddev=self.HMC_eta) for x in theta_0] 
        s_0 = [tf.random.normal((2,), stddev=self.HMC_eta)]
        s_0 = self.__check_trainable(s_0)
        r, s = r_0.copy(), s_0.copy()
        theta = theta_0.copy()
        sigma = sigma_0.copy()
        for _ in range(self.HMC_L):
            theta, sigma, r, s = self.__leapfrog_step(theta, sigma, r, s, self.HMC_dt)
        return self.__accept_reject((theta_0,theta), (sigma_0,sigma), (r_0,r), (s_0,s))

    def select_thetas(self, thetas_train, sigmas_train):
        """ Compute burn-in and samples with skip """
        self.selected = self.selected[self.burn_in::self.skip]
        thetas_train = thetas_train[self.burn_in::self.skip]
        sigmas_train = sigmas_train[self.burn_in::self.skip]
        return thetas_train, sigmas_train

    def train_log(self):
        """ Report log of the training"""
        print('End training:')
        self.compute_time()
        accepted = sum(self.selected)
        rejected = len(self.selected) - accepted
        print(f"\tAccepted Values: {accepted}/{accepted+rejected} {100*accepted/(accepted+rejected) :1.2f}%")
