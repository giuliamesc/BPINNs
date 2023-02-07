from ..template import Laplace1D
from dataclasses import dataclass
import numpy as np

@dataclass
class Laplace1D_cos(Laplace1D):
    name    = "Laplace1D_cos"
    physics = {"diffusion" : 1.}
    # Specifications on the mesh: mesh type and resolutions
    mesh = {
        "mesh_type": "sobol",
         "test_res": 128,
        "inner_res": 128,
        "outer_res": 1
    }
    # Boundaries of the domains
    domains = {
        "sol": [[(0.,1.)]],
        "par": [[(0.,1.)]],
        "full" : [(0.,1.)]
    }
    # Lambda expression of the solution and the parametric field
    @property
    def values(self):
        mu = self.physics["diffusion"]
        return {"u": lambda x: [np.cos(x[0]*8)],
                "f": lambda x: [mu*64*np.cos(x[0]*8)]}

@dataclass
class Laplace1D_sin(Laplace1D):
    name    = "Laplace1D_sin"
    physics = {"diffusion" : 1.}
    # Specifications on the mesh: mesh type and resolutions
    mesh = {
        "mesh_type": "sobol",
         "test_res": 128,
        "inner_res": 64,
        "outer_res": 1
    }
    # Boundaries of the domains
    domains = {
        "sol": [[(-0.8,-0.2)],[(0.2,0.8)]],
        "par": [[(-1.,1.)]],
        "full" : [(-1.,1)]
    }
    # Lambda expression of the solution and the parametric field
    @property
    def values(self):
        mu = self.physics["diffusion"]
        return {"u": lambda x: [np.sin(x[0]*6)**3],
                "f": lambda x: [mu*108*np.sin(6*x[0])*(-2*np.cos(6*x[0])**2 + np.sin(6*x[0])**2)]}
