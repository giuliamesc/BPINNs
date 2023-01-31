from ..template import Oscillator1D_Base
from dataclasses import dataclass
import numpy as np

@dataclass
class Oscillator1D(Oscillator1D_Base):
    name    = "Oscillator1D"
    physics = {"delta" : 2.,
               "omega" : 8}
    # Specifications on the mesh: mesh type and resolutions
    mesh = {
        "mesh_type": "sobol",
         "test_res": 128,
        "inner_res": 64,
        "outer_res": 1
    }
    # Boundaries of the domains
    domains = {
        "sol": [[(0.,0.7)]],
        "par": [[(0.,2.)]],
        "full" : [(0.,2.)]
    }
    # Lambda expression of the solution and the parametric field
    @property
    def values(self):
        A = 1.0
        D = self.physics["delta"]
        Psi = np.sqrt(self.physics["omega"]**2 - self.physics["delta"]**2)
        return {"u": lambda x: [A*np.exp(-D*x[0])*np.sin(Psi*x[0])],
                "f": lambda x: [A*np.exp(-D*x[0])*np.sin(Psi*x[0])]}

