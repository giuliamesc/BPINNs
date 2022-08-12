from abc import ABC, abstractmethod

class Equation(ABC):
    """
    Parent abstract class for pde constraint
    """
    def __init__(self, par):
        """ 
        Constructor
        
        n_input   -> dimension input (1,2 or 3)
        n_out_sol -> dimension of solution
        n_out_par -> dimension of parametric field
        """
        self.n_input   = par.n_input
        self.n_out_sol = par.n_out_sol
        self.n_out_par = par.n_out_par

    @abstractmethod
    def compute_residual(self, x, forward_pass):
        """
        Compute the pde losses, 
        need to be overridden in child classes
        """
        return None

    @abstractmethod
    def pre_process(self, dataset):
        """
        Pre-processes the dataset given as input to the network, 
        need to be overridden in child classes
        """
        return None

    @abstractmethod
    def post_process(self, outputs):
        """
        Transforms back the output of the network, 
        need to be overridden in child classes
        """
        return None
