from .CoreNN import CoreNN
import tensorflow as tf

class PhysNN(CoreNN):

    def __init__(self, par, equation, **kw):
        super(PhysNN, self).__init__(par, **kw)
        self.u_coeff = None  # Store coefficients for denormalizing u in prediction
        self.f_coeff = None  # Store coefficients for denormalizing f in prediction
        self.pinn = equation(par)
        self.inv_flag = par.inv_flag

    @staticmethod
    def tf_convert(tensor): 
        """ Conversion of a numpy array to tensor """
        return tf.cast(tensor, dtype=tf.float32)

    def forward(self, inputs):
        u_tilde = super(PhysNN, self).forward(inputs)
        u = u_tilde[:,:self.n_out_sol]
        f = u_tilde[:,self.n_out_sol:]
        return u, f
