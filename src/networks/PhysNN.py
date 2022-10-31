from .CoreNN import CoreNN
import tensorflow as tf

class PhysNN(CoreNN):

    def __init__(self, par, equation, **kw):
        super(PhysNN, self).__init__(par, **kw)
        self.pinn = equation(par)
        self.inv_flag = par.inv_flag
        self.u_coeff, self.f_coeff = None, None

    @property
    def norm_coeff(self):
        return self.pinn.norm_coeff

    @norm_coeff.setter
    def norm_coeff(self, norm):
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
