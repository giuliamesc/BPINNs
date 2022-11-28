from .template_regression import Regression1D, Regression2D
from dataclasses import dataclass
import numpy as np

@dataclass
class Reg1D_cos(Regression1D):
    name = "reg1D_cos"
    physics = {}
    # Specifications on the domain: mesh type, resolution, boundaries
    analytical_domain = {
        "mesh_type" : "uniform",
        "resolution": 200,
        "domain": [(0,1)]}
    # Lambda expression of the solution and the parametric field
    values = {
        "u": lambda *x: np.cos(x[0]*8),
        "f": lambda *x: np.cos(x[0]*8)}

@dataclass
class Reg1D_sin(Regression1D):
    name = "reg1D_sin"
    physics = {}
    
    mesh = {
        "mesh_type" : "sobol",
         "test_res": 4,
        "inner_res": 16,
        "outer_res": 2
    }
    domains = {
        "sol": [[(-0.8,-0.2)],[(0.2,0.8)]],
        "par": [[(-1.,1.)]],
        "full" : [(-1.,1.)]
    }
    # Lambda expression of the solution and the parametric field
    values = {
        "u": lambda *x: np.sin(x[0]*6)**3,
        "f": lambda *x: np.sin(x[0]*6)**3}