from abc import ABC, abstractmethod

class Equation(ABC):
    """
    Parent abstract class for pde constraint
    """
    def __init__(self, par):
        """ 
        Constructor
        phys_dim dictionary with IO dimension from physical domain
        comp_dim dictionary with IO dimension for NN layers
            n_input   -> dimension input (1,2 or 3)
            n_out_sol -> dimension of solution
            n_out_par -> dimension of parametric field
        """
        self.physics  = par.physics
        self.phys_dim = par.phys_dim
        self.comp_dim = par.comp_dim
        self.phys = dict()
        self.norm = dict()

    @abstractmethod
    def comp_residual(self, inputs, out_sol, out_par, tape):
        return None

