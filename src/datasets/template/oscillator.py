from .template import Data_Config
from dataclasses import dataclass

@dataclass
class Oscillator(Data_Config):
    # Name of the differential equation
    pde = "Oscillator"
    # Parameters to be set in the equation
    physics = {"delta" : float,
               "omega" : float}

@dataclass
class Oscillator1D_Base(Oscillator):
    problem = "Oscillator"
    # Dimensions of the physical domain: dimension of inputs, solution and parametric field
    phys_dom = {
        "n_input"   : 1,
        "n_out_sol" : 1,
        "n_out_par" : 0}
    # Dimensions of the computational domain: dimension of inputs, solution and parametric field
    comp_dom = {
        "n_input"   : 1,
        "n_out_sol" : 1,
        "n_out_par" : 0}
