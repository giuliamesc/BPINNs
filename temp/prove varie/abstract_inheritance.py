from abc import ABC, abstractmethod

class Algorithm(ABC):
    """
    Class for HMC training
    """
    def __init__(self, bayes_nn):
        
        self.model = bayes_nn
        self.loss = list()

    @abstractmethod
    def sample(self):
        pass

class HMC(Algorithm):
    """
    Class for HMC training
    """
    def __init__(self, bayes_nn):
        super(HMC, self).__init__(bayes_nn)

    def sample(self):
        print(self.model)

if __name__ == "__main__":
    hmc = HMC("PROVA")
    hmc.sample()