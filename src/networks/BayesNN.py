from .PredNN import PredNN
from .LossNN import LossNN

class BayesNN(PredNN, LossNN):
    """
    - Initialization of losses dictionaries and equation
    - Contains loss history
    """

    def __init__(self, par, equation):
        
        self.seed    = par.utils["random_seed"] 
        self.history = self.__initialize_losses()
        super(BayesNN, self).__init__(par = par, equation = equation)


    def __initialize_losses(self):
        """ Initializes empty MSE and log-likelihood dictionaries """
        keys = ("Total", "res", "data", "prior")
        loss_dict, logloss_dict = dict(), dict()
        for key in keys: 
            loss_dict[key]    = list()
            logloss_dict[key] = list()
        return (loss_dict, logloss_dict)

    def loss_step(self, new_losses):
        """ Appends new losses to loss history """
        keys = ("Total", "res", "data", "prior")
        for key in keys: 
            self.history[0][key].append(new_losses[0][key])
            self.history[1][key].append(new_losses[1][key])
