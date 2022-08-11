from utility import *

class Laplace(Pde_constraint):
    """
    Laplace pde constraint implementation
    """
    def __init__(self, par, forward_pass):
        super().__init__(par, forward_pass)
        
    def compute_pde_residual(self, x):
        """
        - Laplacian(u) = f -> f + Laplacian(u) = 0
        u shape: (n_sample x n_out_sol)
        f shape: (n_sample x n_out_par)
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u, f = self.forward_pass(x)
            lap = Operators.laplacian_vector(tape, u, x, self.n_out_sol)
        return lap + f
