from dataclasses import dataclass
from abc import ABC

@dataclass
class Data_Config(ABC):

    # Name of the test case
    name = None
    # Name of the differential equation
    pde = None
    # Dimensions of the physical domain: dimension of inputs, solution and parametric field
    phys_dom = {
        "n_input"   : int,
        "n_out_sol" : int,
        "n_out_par" : int}
    # Dimensions of the computational domain: dimension of inputs, solution and parametric field
    comp_dom = {
        "n_input"   : int,
        "n_out_sol" : int,
        "n_out_par" : int}
    # Specifications on the domain: mesh type, resolution, boundaries
    analytical_domain = {
        "mesh_type" : str,
        "resolution": int,
        "domain"    : list}