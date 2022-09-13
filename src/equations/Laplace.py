from .Equation  import Equation
from .Operators import Operators
import tensorflow as tf

class Laplace(Equation):
    """
    Laplace pde constraint implementation
    """
    def __init__(self, par):
        super().__init__(par)
        self.mu = tf.constant(par.physics["diffusion"])

    def parametric_field(self, u_tilde, inputs, tape):
        lap_u = Operators.laplacian_vector(tape, u_tilde, inputs, 1)
        par_f = - self.mu * lap_u
        return par_f

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