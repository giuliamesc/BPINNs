from .PredNN import PredNN
from .LossNN import LossNN
from .equations import Laplace

class BayesNN(PredNN, LossNN):
    """
    - Initialization of losses dictionaries and equation
    - Contains loss history
    """

    def __init__(self, par):
        
        self.seed    = par.utils["random_seed"] 
        self.history = self.__initialize_losses()

        equation  = self.__initialize_equation(par)
        comp_res  = equation.compute_residual
        
        pre_proc  = equation.pre_process
        post_proc = equation.post_process
        comp_proc = equation.comp_process
        data_proc = equation.data_process
        
        super(BayesNN, self).__init__(par=par, comp_res=comp_res,
                                      pre=pre_proc, post=post_proc, 
                                      proc=comp_proc, data=data_proc)

    def __initialize_losses(self):
        """ Initializes empty MSE and log-likelihood dictionaries """
        keys = ("Total", "res", "data", "prior")
        loss_dict, logloss_dict = dict(), dict()
        for key in keys: 
            loss_dict[key]    = list()
            logloss_dict[key] = list()
        return (loss_dict, logloss_dict)

    def __initialize_equation(self, par):
        """ Initializes a class of equation describing the problem """
        equation = par.dataset
        if   equation == "Laplace1D_cos": return Laplace(par)
        elif equation == "Laplace2D_cos": return Laplace(par)
        else: raise("Equation not implemeted!")

    def loss_step(self, new_losses):
        """ Appends new losses to loss history """
        keys = ("Total", "res", "data", "prior")
        for key in keys: 
            self.history[0][key].append(new_losses[0][key])
            self.history[1][key].append(new_losses[1][key])
