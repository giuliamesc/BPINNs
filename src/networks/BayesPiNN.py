import CoreNN
import LossNN

class BayesPiNN(CoreNN, LossNN):
    """
    **** INHERITANCE ****
    - CoreNN: nn_params
    - CoreNN: forward()
    - LossNN:
    """

    def __init__(self, par):
        self.bayes_nn = LossNN(par, CoreNN(par))

if __name__ == "__main__":
    prova = BayesPiNN()
    pass