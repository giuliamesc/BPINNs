import numpy as np
import tensorflow as tf

from algorithms.Algorithm import Algorithm

class HMC(Algorithm):
    """
    Class for HMC training
    """
    def __init__(self, bayes_nn, param_method):
        super().__init__(bayes_nn, param_method)
        self.HMC_L  = param_method["HMC_L"]
        self.HMC_dt = param_method["HMC_dt"]
        self.eta = 1.0

    def sample_theta(self, theta_0):
        
        r_0 = [tf.random.normal(x.shape) for x in theta_0]

        r     = r_0.copy()
        theta = theta_0.copy()
        
        for _ in range(self.HMC_L):
            theta, r = self.__leapfrog(theta, r, self.HMC_dt)

        #print(theta)
        #return theta
        return self.__accept_reject(theta_0, theta, r_0, r)

    def __leapfrog(self, old_theta, r, dt):
        r = [ x - y * dt/2 for x,y in zip(r, self.model.grad_loss(self.data))]
        new_theta = [ x + y * dt for x,y in zip(old_theta, r)]
        self.model.nn_params = new_theta 
        r = [ x - y * dt/2 for x,y in zip(r, self.model.grad_loss(self.data))]
        return new_theta, r

    def __accept_reject(self, theta_0, theta_1, r_0, r_1):

        h0 = self.__hamiltonian(theta_0, r_0)
        h1 = self.__hamiltonian(theta_1, r_1)
        print(f"\th0: {h0 :1.3e}")
        print(f"\th1: {h1 :1.3e}")
        print(f"\t h: {h1-h0 :1.3e}")
        #alpha = min(1, np.exp(h1-h0))
        log_alpha = min(0, -h1+h0)

        p = np.random.uniform()
        log_p = np.log(p)
        
        #accept = alpha <= p
        accept = log_alpha <= log_p
        theta  = theta_1 if accept else theta_0

        print("ACCEPT" if accept else "REJECT")
        
        return theta 

    def __hamiltonian(self, theta, r):
        self.model.nn_params = theta
        u = self.model.loss_total(self.data)[1]["Total"].numpy()
        v = sum([tf.norm(t).numpy()**2 for t in r])
        v *= self.eta**2/2
        return u + v

