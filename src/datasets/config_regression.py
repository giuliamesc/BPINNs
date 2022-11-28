from .template_regression import Regression1D, Regression2D
from dataclasses import dataclass
import numpy as np

@dataclass
class Reg1D_cos(Regression1D):
    name = "reg1D_cos"
    physics = {}
    # Specifications on the mesh: mesh type and resolutions
    mesh = {
        "mesh_type": "sobol",
         "test_res": 128,
        "inner_res": 64,
        "outer_res": 8
    }
    # Boundaries of the domains
    domains = {
        "sol": [[(0,2),(1,3)],[(4,6),(1,3)]],
        "par": [[(2,5),(0,1)]],
        "full" : [(0,6),(0,3)]
    }
    # Lambda expression of the solution and the parametric field
    values = {
        "u": lambda *x: np.cos(x[0]*8),
        "f": lambda *x: np.cos(x[0]*8)}

@dataclass
class Reg1D_sin(Regression1D):
    name = "reg1D_sin"
    physics = {}
    # Specifications on the mesh: mesh type and resolutions
    mesh = {
        "mesh_type": "sobol",
         "test_res": 4,
        "inner_res": 16,
        "outer_res": 4
    }
    # Boundaries of the domains
    domains = {
        "sol": [[(-0.8,-0.2)],[(0.2,0.8)]],
        "par": [[(-1.,1.)]],
        "full" : [(-1.,1)]
    }
    # Lambda expression of the solution and the parametric field
    values = {
        "u": lambda *x: np.sin(x[0]*6)**3,
        "f": lambda *x: np.sin(x[0]*6)**3}