from abc import ABC, abstractmethod
from tqdm import tqdm
import time, datetime
import os

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
        self.model.process_params = self.model.pinn.comp_process(dataset)
        self.data = self.model.pinn.data_process(dataset, self.model.process_params)

    def __train_step(self, epoch):

        # Sampling new theta
        match type(self).__name__:
            case "TEST": new_theta, new_sigma = self.sample_theta(epoch)
            case "HMC" : new_theta, new_sigma = self.sample_theta(self.model.nn_params, self.model.sg_params)
            case "SVGD": new_theta, new_sigma = self.sample_theta()
            case "VI"  : new_theta, new_sigma = self.sample_theta()
            case _: raise Exception("Method not Implemented!")
        # Saving new Theta
        self.model.nn_params = new_theta
        self.model.sg_params = new_sigma
        # Computing History
        loss, logloss = self.model.loss_total(self.data)
        self.model.loss_step((loss,logloss))
        
        return new_theta, new_sigma

    def __train_loop(self, epochs):
        epochs_loop = range(epochs)
        if not self.debug_flag: 
             epochs_loop = tqdm(epochs_loop)
             epochs_loop.ncols=max(75,int(os.get_terminal_size().columns/2))
             epochs_loop.set_description_str("Training Progress")
        return epochs_loop

    def train(self):

        # Store thetas in this round of training
        thetas_train = list()
        sigmas_train = list()
        # Sampling new thetas
        self.epochs_loop = self.__train_loop(self.epochs) 
        for i in self.epochs_loop:
            if self.debug_flag: print(f'  START EPOCH {i+1}')
            step = self.__train_step(i)
            thetas_train.append(step[0])
            sigmas_train.append(step[1])
    
        # Select which thetas must be saved
        thetas_train, sigmas_train = self.select_thetas(thetas_train, sigmas_train)
        # Save thetas in the bnn
        self.model.thetas += thetas_train
        self.model.sigmas += sigmas_train
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
