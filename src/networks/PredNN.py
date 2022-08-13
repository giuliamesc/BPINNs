from .CoreNN import CoreNN

class PredNN(CoreNN):
    
    def __init__(self, pre, post, **kw):
        
        super(PredNN, self).__init__(**kw)

        # Functions for pre and post processing of inputs and outputs of the network
        self.pre_process = pre
        self.post_process = post
        
        # Empty list where samples of network parameters will be stored
        self.thetas = list() 

    def __compute_sample(self, theta, inputs):

        self.nn_params = theta
        sample = self.model.forward(inputs, split=True)

        return self.post_process(sample)
    
    def __compute_all_samples(self, inputs):

        inputs = self.pre_process(inputs)
        out_sol = list()
        out_par = list()
        
        for theta in self.thetas:
            outputs = self.__compute_sample(theta, inputs)
            out_sol.append(outputs[0])
            out_par.append(outputs[1])

        return out_sol, out_par

    def compute_errors(self):
        return None

    def mean_and_std(self):
        return None

    def draw_samples(self):
        return None

    def show_errors(self, errors):
        pass