from abc import ABC, abstractmethod
import time, datetime

class Algorithm(ABC):
    """
    Class for HMC training
    """
    def __init__(self, bayes_nn, param_method):
        
        self.t0 = time.time() 
        self.model  = bayes_nn
        self.params = param_method
        self.epochs = self.params["epochs"]

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

    def train(self):
        
        for i in range(self.epochs):

            print(f'Epoch {i+1}')

            grad, loss, logloss = self.model.grad_loss(self.data)
            self.model.loss_step((loss,logloss))

            new_theta = self.sample_theta(i, grad)
            self.model.nn_params = new_theta
            self.model.thetas.append(new_theta)

    @abstractmethod
    def sample_theta(self):
        return None
