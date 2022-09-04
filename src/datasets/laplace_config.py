from .laplace_template import Laplace1D, Laplace2D
from dataclasses import dataclass
import numpy as np

@dataclass
class Laplace1D_cos(Laplace1D):
    name    = "Laplace1D_cos"
    physics = {"diffusion" : 1.}
    # Specifications on the domain: mesh type, resolution, boundaries
    analytical_domain = {
        "mesh_type" : "sobol",
        "resolution": 200,
        "domain": [(0,1)]}
    # Lambda expression of the solution and the parametric field
    analytical_solution = {
        "u": lambda *x: np.cos(x[0]*8),
        "f": lambda *x: 64*np.cos(x[0]*8)}