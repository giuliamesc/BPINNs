import tensorflow as tf

class pde_constraint:
    """
    Class for the pde_constraint
    """

    def __init__(self, bayes_methods, n_input, n_output_vel):
        """! Constructor
        @param bayes_methods name of method used
        @param n_input dimension input (1,2 or 3)
        @param n_output_vel dimension of velocity output (1 isotropic, >1 anisotropic)"""
        self.bayes_methods = bayes_methods # "SVGD", "HMC", ...
        self.n_input = n_input  # 1,2,3 (1D, 2D, 3D)
        self.n_output_vel = n_output_vel

    def compute_pde_losses(self, u_gr, f_gr, f):
        """compute the pde losses, need to be override in child classes"""
        return 0.

class laplace(pde_constraint):
    def __init__(self, par):

        super().__init__(par.method, par.n_input, par.n_output_vel)

    def compute_pde_losses(self, u_gr_2, f):
        """
        - Laplacian(u) = f -> f + Laplacian(u)=0
        """
        loss_1 = tf.reduce_sum(f[:,:,:self.n_input])
        for u_gr_2_comp in u_gr_2:
            loss_1 += u_gr_2_comp

        return loss_1
