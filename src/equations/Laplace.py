from .Equation  import Equation
from .Operators import Operators

class Laplace(Equation):
    """
    Laplace pde constraint implementation
    """
    def __init__(self, par):
        super().__init__(par)
        self.phys_coeff = par.physics["diffusion"]

    @property
    def phys_coeff(self):
        return self.phys

    @phys_coeff.setter
    def phys_coeff(self, phys):
        self.phys["mu"] = phys

    @property
    def norm_coeff(self):
        return self.norm

    @norm_coeff.setter
    def norm_coeff(self, norm):
        self.norm["u_std"]   = norm[0][1]
        self.norm["f_std"]   = norm[1][1]
        self.norm["u_mean"]  = norm[0][0]
        self.norm["f_mean"]  = norm[1][0]
        self.norm["mu_norm"] = self.norm["u_std"]/self.norm["f_std"]

    def comp_residual(self, inputs, out_sol, out_par, tape):
        u_list = Operators.tf_unpack(out_sol)
        lap_u  = Operators.laplacian_vector(tape, u_list, inputs)
        lap_u  = Operators.tf_pack(lap_u)
        lhs = - lap_u * self.norm["mu_norm"] * self.phys["mu"]
        rhs = out_par + self.norm["f_mean"]/self.norm["f_std"]
        return lhs - rhs