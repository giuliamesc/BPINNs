from abc import ABC, abstractmethod
from tqdm import tqdm
import time, datetime

class Algorithm(ABC):
    """
    Class for HMC training
    """
    def __init__(self, bayes_nn, param_method, debug_flag):
        
        self.t0 = time.time() 
        self.model  = bayes_nn
        self.params = param_method
        self.epochs = self.params["epochs"]
        self.debug_flag = debug_flag

    def compute_time(self):
        training_time  = time.time() - self.t0
        formatted_time  = str(datetime.timedelta(seconds=int(training_time)))
        print(f'\tFinished in {formatted_time}')
        
    @property
    def data_train(self):
        return self.data

    @data_train.setter
    def data_train(self, dataset):
        processed_data = self.model.pre_process(dataset)
        self.data = processed_data

    def __train_step(self, epoch):

        # Sampling new theta
        match type(self).__name__:
            case "TEST": new_theta = self.sample_theta(epoch)
            case "HMC" : new_theta = self.sample_theta(self.model.nn_params)
            case "SVGD": new_theta = self.sample_theta()
            case "VI"  : new_theta = self.sample_theta()
            case _: raise Exception("Method not Implemented!")
        # Saving new Theta
        self.model.nn_params = new_theta
        # Computing History
        loss, logloss = self.model.loss_total(self.data)
        self.model.loss_step((loss,logloss))
        
        return new_theta

    def train(self):

        # Store thetas in this round of training
        thetas_train = list()
        
        # Sampling new thetas
        if self.debug_flag :
            for i in range(self.epochs):
                print(f'START EPOCH {i+1}')
                thetas_train.append(self.__train_step(i))
        else:
            for i in tqdm(range(self.epochs)):
                thetas_train.append(self.__train_step(i))
    
        # Select whict thetas must be saved
        thetas_train = self.select_thetas(thetas_train)
        # Save thetas in the bnn
        self.model.thetas += thetas_train
        # Report training information
        self.train_log()

    def train_log(self):
        """ Report log of the training"""
        print('End training:')
        self.compute_time()

    @abstractmethod
    def sample_theta(self):
        """ 
        Method for sampling a single new theta
        Must be overritten in child classes
        """
        return None

    @abstractmethod
    def select_thetas(self, thetas_train):
        """ 
        Compute burn-in and skip samples
        Must be overritten in child classes
        """
        return list()
