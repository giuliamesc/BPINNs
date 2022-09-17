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

    @abstractmethod
    def solution(self, u_tilde, inputs, tape=None):
        return None

    @abstractmethod
    def parametric_field(self, u_tilde, inputs, tape):
        return None
    
    @abstractmethod
    def comp_process(self, dataset):
        """
        Extracts the relevant features from the dataset to perform pre-processing, 
        need to be overridden in child classes
        """
        return None

    @abstractmethod
    def data_process(self, dataset, params):
        """
        Builds the pre-processed dataset, 
        need to be overridden in child classes
        """
        return None

    @abstractmethod
    def pre_process(self, inputs, params):
        """
        Pre-processes the dataset given as input to the network, 
        need to be overridden in child classes
        """
        return None

    @abstractmethod
    def post_process(self, outputs, params):
        """
        Transforms back the output of the network, 
        need to be overridden in child classes
        """
        return None
