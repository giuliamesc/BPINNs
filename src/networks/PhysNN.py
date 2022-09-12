from .CoreNN import CoreNN
import tensorflow as tf

class PhysNN(CoreNN):

    def __init__(self, par, equation, **kw):
        super(PhysNN, self).__init__(par, **kw)
        
        self.pinn = equation(par)
        self.n_out_par = par.comp_dim.n_out_par
        self.inv_flag  = par.inv_flag

        # Sigmas Operations -> Lambda
        self.sg_params = [self.tf_convert([par.sigmas["data_pn"], par.sigmas["pde_pn"]])]
        self.sg_flags  = [par.sigmas["data_pn_flag"], par.sigmas["pde_pn_flag"]]
        self.sigmas = list()

    @staticmethod
    def tf_convert(tensor): 
        """ Conversion of a numpy array to tensor """
        return tf.cast(tensor, dtype=tf.float32)

    def forward(self, inputs):
        u_tilde = super(PhysNN, self).forward(inputs)
        u = u_tilde
        f = self.pinn.parametric_field(u_tilde)
        return u, f