from .PredNN import PredNN
from .LossNN import LossNN

class BayesNN(PredNN, LossNN):
    """
    - Initialization of losses dictionaries and equation
    - Contains loss history
    """

    def __init__(self, par, equation):
        
        super(BayesNN, self).__init__(par = par, equation = equation)
        self.seed    = par.utils["random_seed"]
        self.history = self.__initialize_losses()


    def __initialize_losses(self):
        """ Initializes empty MSE and log-likelihood dictionaries """
        loss_dict, logloss_dict = dict(), dict()
        for key in self.keys: 
            loss_dict[key]    = list()
            logloss_dict[key] = list()
        return (loss_dict, logloss_dict)

    def loss_step(self, new_losses):
        """ Appends new losses to loss history """
        for key in self.keys: 
            self.history[0][key].append(new_losses[0][key])
            self.history[1][key].append(new_losses[1][key])
