from .template_laplace import Laplace1D, Laplace2D
from dataclasses import dataclass
import numpy as np

@dataclass
class Laplace1D_cos(Laplace1D):
    name    = "Laplace1D_cos"
    physics = {"diffusion" : 1.}
    # Specifications on the domain: mesh type, resolution, boundaries
    analytical_domain = {
        "mesh_type" : "uniform",
        "resolution": 200,
        "domain": [(0,1)]}
    # Lambda expression of the solution and the parametric field
    analytical_solution = {
        "u": lambda *x: np.cos(x[0]*8),
        "f": lambda *x: np.cos(x[0]*8)}
@dataclass
class Laplace1D_sin(Laplace1D):
    name    = "Laplace1D_sin"
    physics = {"diffusion" : 1/64}
    # Specifications on the domain: mesh type, resolution, boundaries
    analytical_domain = {
        "mesh_type" : "uniform",
        "resolution": 200,
        "domain": [(-0.7,0.7)]}
    # Lambda expression of the solution and the parametric field
    analytical_solution = {
        "u": lambda *x: np.sin(x[0]*6)**3,
        "f": lambda *x: 1e-2*108*np.sin(x[0]*6)*(2*np.cos(x[0]*6)**2-np.sin(x[0]*6)**2)}
"""
@dataclass
class Laplace1D_sin(Laplace1D):
    name    = "Laplace1D_sin"
    physics = {"diffusion" : 2.}
    # Specifications on the domain: mesh type, resolution, boundaries
    analytical_domain = {
        "mesh_type" : "uniform",
        "resolution": 200,
        "domain": [(0,1)]}
    # Lambda expression of the solution and the parametric field
    analytical_solution = {
        "u": lambda *x: 1-x[0]*np.exp(-x[0]),
        "f": lambda *x: 0.5*x[0]*np.exp(-x[0])-np.exp(-x[0])}
"""