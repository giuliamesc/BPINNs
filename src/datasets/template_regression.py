from .template_general import Data_Config
from dataclasses import dataclass

@dataclass
class Regression(Data_Config):
    pde = "regression"
    # Parameters to be set in the equation
    physics = dict()

@dataclass
class Regression1D(Regression):
    problem = "Regression"
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
class Regression2D(Regression):
    problem = "Regression"
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