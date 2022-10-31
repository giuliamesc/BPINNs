from .CoreNN import CoreNN
import tensorflow as tf

class PhysNN(CoreNN):

    def __init__(self, par, equation, **kw):
        super(PhysNN, self).__init__(par, **kw)
        self.norm = None # Store coefficients for denormalizing u, f in prediction
        self.pinn = equation(par)
        self.inv_flag = par.inv_flag

    @property
    def norm_coeff(self): 
        return self.norm

    @norm_coeff.setter
    def norm_coeff(self, norm):
        self.norm = norm
        self.pinn.norm_coeff = norm

    @staticmethod
    def tf_convert(tensor): 
        """ Conversion of a numpy array to tensor """
        return tf.cast(tensor, dtype=tf.float32)

    def forward(self, inputs):
        u_tilde = super(PhysNN, self).forward(inputs)
        u = u_tilde[:,:self.n_out_sol]
        f = u_tilde[:,self.n_out_sol:]
        return u, f
