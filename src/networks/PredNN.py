import CoreNN

class PredNN(CoreNN):
    
    def __init__(self, pre, post, **kw):
        
        super(PredNN, self).__init__(**kw)

        # Functions for pre and post processing of inputs and outputs of the network
        self.pre_process = pre
        self.post_process = post
        
        # Empty list where samples of network parameters will be stored
        self.thetas = list() 

    def predict(self):
        pass

    def compute_errors(self):
        pass

    def mean_and_std(self):
        pass
