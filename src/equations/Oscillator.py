from .Equation  import Equation
from .Operators import Operators

class Oscillator(Equation):
    """
    Laplace pde constraint implementation
    """
    def __init__(self, par):
        super().__init__(par)
        self.phys_coeff = (par.physics["delta"], par.physics["omega"]) 

    @property
    def phys_coeff(self):
        return self.phys

    @phys_coeff.setter
    def phys_coeff(self, phys):
        self.phys["delta"] = phys[0]
        self.phys["omega"] = phys[1]

    @property
    def norm_coeff(self):
        return self.norm

    @norm_coeff.setter
    def norm_coeff(self, norm):
        self.norm["u_std"]   = norm["sol_std"]
        self.norm["u_mean"]  = norm["sol_mean"]

    def comp_residual(self, inputs, out_sol, _, tape):
        x_list = Operators.tf_unpack(out_sol)
        x_tt  = Operators.tf_pack(Operators.laplacian_vector(tape, x_list, inputs))
        x_t   = Operators.tf_pack(Operators.divergence_vector(tape, x_list, inputs))
        x     = Operators.tf_pack(x_list)
        d, w = self.phys["delta"], self.phys["omega"]
        return x_tt + 2*d*x_t + w*x