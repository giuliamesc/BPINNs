from utility import compute_gui_len
from setup import BatcherDataset
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
        self.curr_ep = 0

    def compute_time(self):
        training_time  = time.time() - self.t0
        formatted_time   = str(datetime.timedelta(seconds=int(training_time)))
        print(f'\tFinished in {formatted_time}')
        
    @property
    def data_train(self):
        return self.__data

    @data_train.setter
    def data_train(self, dataset):
        self.model.u_coeff = dataset.norm_coeff["sol_mean"], dataset.norm_coeff["sol_std"]
        self.model.f_coeff = dataset.norm_coeff["par_mean"], dataset.norm_coeff["par_std"]
        self.__data = dataset

    def __train_step(self, epoch):
        next(self.data_batch) 
        match type(self).__name__:
            case "TEST": new_theta = self.sample_theta(epoch)
            case "ADAM": new_theta = self.sample_theta(self.model.nn_params)
            case "HMC" : new_theta = self.sample_theta(self.model.nn_params)
            case "SVGD": new_theta = self.sample_theta() # OLD THETA (SPACE LIST)
            case "VI"  : new_theta = self.sample_theta()
            case _: raise Exception("Method not Implemented!")
        self.update_history(new_theta) 
        return new_theta

    def __train_loop(self, epochs):
        epochs_loop = range(epochs)
        if not self.debug_flag: 
             epochs_loop = tqdm(epochs_loop)
             epochs_loop.ncols=compute_gui_len()
             epochs_loop.set_description_str("Training Progress")
        return epochs_loop

    def train(self):

        # Store thetas in this round of training
        thetas_train = list()
        # Normalizing dataset
        self.data_train.normalize_dataset()
        self.model.norm_coeff = self.data_train.norm_coeff
        self.data_batch = BatcherDataset(self.data_train, num_batch=1)

        # Sampling new thetas
        self.epochs_loop = self.__train_loop(self.epochs) 
        for i in self.epochs_loop:
            if self.debug_flag: print(f'  START EPOCH {i+1}')
            self.curr_ep = i+1
            step = self.__train_step(i)
            thetas_train.append(step)
    
        # Denormalizing dataset
        self.data_train.denormalize_dataset()
        # Select which thetas must be saved
        thetas_train = self.select_thetas(thetas_train)
        # Save thetas in the bnn
        self.model.thetas += thetas_train

        # Report training information
        self.train_log()

    def train_log(self):
        """ Report log of the training"""
        print('End training:')
        self.compute_time()

    def update_history(self, new_theta):
        # Saving new Theta
        self.model.nn_params = new_theta
        # Computing History
        pst, llk = self.model.metric_total(self.data_batch)
        self.model.loss_step((pst,llk))

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
