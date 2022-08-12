import PredNN
import LossNN

from equations import Laplace

class BayesNN(PredNN, LossNN):
    """
    **** INHERITANCE ****
    - CoreNN: nn_params
    - CoreNN: forward()
    - LossNN:
    - PredNN:
    """

    def __init__(self,par):
        
        equation = self.__initialize_equation(par)
        
        comp_res  = equation.compute_residual
        pre_proc  = equation.pre_process
        post_proc = equation.post_process
        
        super(BayesNN,self).__init__(par=par, comp_res=comp_res,
                                     pre_proc=pre_proc, post_proc=post_proc)
        

    def __initialize_equation(par):
        equation = par.experiment["dataset"]
        if   equation == "laplace1D_cos": return Laplace(par)
        elif equation == "laplace2D_cos": return Laplace(par)
        else: raise("Equation not implemeted!")

    def save_network(self, filepath):
        pass

    def load_network(self, filepath):
        pass

if __name__ == "__main__":
    prova = BayesNN()
