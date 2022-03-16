import tensorflow as tf

class pde_constraint:
    """
    Class for the pde_constraint
    """

    def __init__(self, n_input, n_out_sol, n_out_par):
        """! Constructor
        @param n_input dimension input (1,2 or 3)
        @param n_out_sol dimension of solution
        @param n_out_par dimension of parametric field
        """
        
        self.n_input = n_input
        self.n_out_sol = n_out_sol
        self.n_out_par = n_out_par

    def compute_pde_losses(self, u_gr, f_gr, f):
        """compute the pde losses, need to be override in child classes"""
        return 0.

class laplace(pde_constraint):
    def __init__(self, par):

        super().__init__(par.n_input, par.n_out_sol, par.n_out_par)

    def compute_pde_losses(self, u_gr_2, f):
        """
        - Laplacian(u) = f -> f + Laplacian(u) = 0
        """
        # compute the loss 1: residual of the PDE

        loss_1 = tf.expand_dims(tf.reduce_sum(f, axis=-1),axis=-1)

        for u_gr_2_comp in u_gr_2:
            loss_1 += u_gr_2_comp
        
        return loss_1
