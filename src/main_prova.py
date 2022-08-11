from equations import Laplace
from networks import LossNN

class TempPar:
    def __init__(self):
        
        self.n_inputs  = 1
        self.n_out_sol = 1
        self.n_out_par = 1

        self.architecture = {
            "activation": "swish",
            "n_layers" : 3,
            "n_neurons": 5
        },

        self.pde = "laplace"
        
par = TempPar()
prova = LossNN(par)
print("Done")