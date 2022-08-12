from abc import ABC, abstractmethod
import time, datetime

class Algorithm(ABC):
    """
    Class for HMC training
    """
    def __init__(self, bayes_nn, dataset):
        
        self.t0 = time.time() 
        self.data_train = dataset
        self.model = bayes_nn
        self.loss = list()

    def compute_time(self):
        
        training_time = time.time() - self.t0
        formatted_time = str(datetime.timedelta(seconds=int(training_time)))
        print('Finished in', formatted_time)
        
        return formatted_time
        
    @property
    def data_train(self):
        return self.data

    @data_train.setter
    def data_train(self, dataset):
        processed_data = self.model.pre_process(dataset)
        self.data = processed_data


    def train(self, par):
        n_epochs = par
        for _ in range(n_epochs):
            new_theta = self.sample()
            self.bayes_nn.theta.append(new_theta)

    @abstractmethod
    def sample(self):
        return None
