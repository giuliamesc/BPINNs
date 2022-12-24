from .HMC import HMC
from .VI import VI
from .SVGD import SVGD
from .ADAM import ADAM

class Trainer():
    def __init__(self, bayes_nn, params, dataset):
        self.debug_flag  = params.utils["debug_flag"]
        self.model = bayes_nn
        self.params = params
        self.dataset = dataset

    def __switch_algorithm(self, method):
        """ Returns an instance of the class corresponding to the selected method """
        match method:
            case "ADAM": return ADAM
            case "HMC" : return HMC
            case "SVGD": return SVGD
            case "VI"  : return VI
            case _ : raise Exception("This algorithm does not exist!")

    def __algorithm(self, method, par_method):
        algorithm = self.__switch_algorithm(method)
        algorithm = algorithm(self.model, par_method, self.debug_flag)
        algorithm.data_train = self.dataset
        return algorithm

    def pre_train(self):
        if self.params.init is None: return
        print(f"Pre-training phase with method {self.params.init}...")
        alg = self.__algorithm(self.params.init, self.params.param_init)
        alg.train()
        self.model.nn_params = self.model.thetas.pop()
        
    def train(self):
        print(f"Training phase with method {self.params.method}...")
        alg = self.__algorithm(self.params.method, self.params.param_method)
        alg.train()

