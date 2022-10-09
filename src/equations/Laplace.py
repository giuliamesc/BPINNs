from .Equation  import Equation
from .Operators import Operators

class Laplace(Equation):
    """
    Laplace pde constraint implementation
    """
    def __init__(self, par):
        super().__init__(par)
        self.mu = par.physics["diffusion"]

    def comp_residual(self, inputs, out_sol, out_par, tape):
        u_list = Operators.tf_unpack(out_sol)
        lap_u  = Operators.laplacian_vector(tape, u_list, inputs)
        lap_u  = Operators.tf_pack(lap_u)
        return lap_u * self.mu + out_par

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