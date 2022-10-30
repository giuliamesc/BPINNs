from .Equation  import Equation
from .Operators import Operators

class Laplace(Equation):
    """
    Laplace pde constraint implementation
    """
    def __init__(self, par):
        super().__init__(par)
        self.mu = Operators.tf_convert(par.physics["diffusion"])

    def comp_residual(self, inputs, out_sol, out_par, tape):
        u_list = Operators.tf_unpack(out_sol)
        lap_u  = Operators.laplacian_vector(tape, u_list, inputs)
        lap_u  = Operators.tf_pack(lap_u)
        return lap_u * self.mu + out_par

    def _normalize_data(self, vec):
        u_span = max(vec[1])-min(vec[1])
        f_span = max(vec[2])-min(vec[2])
        new_dom_data = (vec[0], vec[1]/u_span, vec[2]/f_span)
        return new_dom_data