from .data_template import Data_Config
from dataclasses import dataclass

@dataclass
class Laplace(Data_Config):
    # Name of the differential equation
    pde = "laplace"
    # Parameters to be set in the equation
    physics = {"diffusion" : int}

@dataclass
class Laplace1D(Laplace):
    # Dimensions of the physical domain: dimension of inputs, solution and parametric field
    phys_dom = {
        "n_input"   : 1,
        "n_out_sol" : 1,
        "n_out_par" : 1}
    # Dimensions of the computational domain: dimension of inputs, solution and parametric field
    comp_dom = {
        "n_input"   : 1,
        "n_out_sol" : 1,
        "n_out_par" : 1}

@dataclass
class Laplace2D(Laplace):
    # Dimensions of the physical domain: dimension of inputs, solution and parametric field
    phys_dom = {
        "n_input"   : 2,
        "n_out_sol" : 1,
        "n_out_par" : 1}
    # Dimensions of the computational domain: dimension of inputs, solution and parametric field
    comp_dom = {
        "n_input"   : 2,
        "n_out_sol" : 1,
        "n_out_par" : 1}