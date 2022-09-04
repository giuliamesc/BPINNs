from .Equation  import Equation
from .Operators import Operators
import tensorflow as tf

class Laplace(Equation):
    """
    Laplace pde constraint implementation
    """
    def __init__(self, par):
        super().__init__(par)
        self.mu = par.physics["diffusion"]
        
    def compute_residual(self, x, forward_pass):
        """
        - Laplacian(u) = f -> f + Laplacian(u) = 0
        u shape: (n_sample x n_out_sol)
        f shape: (n_sample x n_out_par)
        """
        x = tf.convert_to_tensor(x)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u, f = forward_pass(x, split = True)
            lap = Operators.laplacian_vector(tape, u, x, self.comp_dim.n_out_sol)
        return self.mu * lap + f

    def comp_process(self, dataset):
        params = dict()
        return params

    def data_process(self, dataset, params):
        """ TO BE DONE """
        new_dataset = dataset
        return new_dataset

    def pre_process(self, inputs, params):
        """ Pre-process in Laplace problem is the identity transformation """
        return inputs

    def post_process(self, outputs, params):
        """ Post-process in Laplace problem is the identity transformation """
        return outputs