from utility import *

class Laplace(Pde_constraint):
    """
    Laplace pde constraint implementation
    """
    def __init__(self, par):
        super().__init__(par)
        
    def compute_pde_residual(self, x, forward_pass):
        """
        - Laplacian(u) = f -> f + Laplacian(u) = 0
        u shape: (n_sample x n_out_sol)
        f shape: (n_sample x n_out_par)
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u, f = forward_pass(x)
            lap = Operators.laplacian_vector(tape, u, x, self.n_out_sol)
        return lap + f

    def pre_process(self, dataset):
        return dataset

    def post_process(self, outputs):
        return outputs