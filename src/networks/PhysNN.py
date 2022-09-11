from .CoreNN import CoreNN

class PhysNN(CoreNN):

    def __init__(self, par, equation, **kw):
        super(PhysNN, self).__init__(par, **kw)
        
        self.pinn = equation(par)
        self.n_out_par = par.comp_dim.n_out_par

    def forward(self, inputs):
        u_tilde = super(PhysNN, self).forward(inputs)
        u = u_tilde
        f = u_tilde # To be updated
        return u, f