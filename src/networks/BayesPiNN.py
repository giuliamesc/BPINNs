import PredNN
import LossNN

class BayesPiNN(PredNN, LossNN):
    """
    **** INHERITANCE ****
    - CoreNN: nn_params
    - CoreNN: forward()
    - LossNN:
    """

    def __init__(self, par, problem):
        #self.bayes_nn = CoreNN(par)
        #self.bayes_nn = LossNN(par, CoreNN(par))
        self.problem = problem # istance of the class correspondent to the problem
        self.data_train = None

    def preprocess(self):
        self.data_train = self.problem.preprocess(self.data_train)

if __name__ == "__main__":
    prova = BayesPiNN()
    pass