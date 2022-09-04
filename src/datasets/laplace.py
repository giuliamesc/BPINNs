from dataclasses import dataclass
from abc import ABC
import numpy as np

@dataclass
class Laplace1D(ABC):

    # Name of the differential equation
    pde = "laplace"
    # Parameters to be set in the equation
    physics = {"diffusion" : 1.}
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
    # Specifications on the domain: mesh type, resolution, boundaries
    analytical_domain = {
        "mesh_type" : "",
        "resolution": 0,
        "domain": [(0,0)]}
    # Lambda expression of the solution and the parametric field
    analytical_solution = { 
        "u": lambda *x : 0*x,
        "f": lambda *x : 0*x}

@dataclass
class Laplace1D_cos(Laplace1D):

    name = "Laplace1D_cos"
    physics = {"diffusion" : 1.}
    analytical_domain = {
        "mesh_type" : "sobol",
        "resolution": 200,
        "domain": [(0,1)]}
    analytical_solution = {
        "u": lambda *x: np.cos(x[0]*8),
        "f": lambda *x: 64*np.cos(x[0]*8)}