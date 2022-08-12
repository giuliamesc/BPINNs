class Storage():
    def __init__(self, path_result, path_plot, path_weights):
        self.path_result = path_result
        self.path_plot = path_plot
        self.path_weights = path_weights
        pass

    def save_training(self, theta, loss):
        pass

    def save_results(self, functions_confidence, functions_nn_samples):
        pass

    def save_errors(self, errors):
        pass

    def load_losses():
        pass
    
    def load_confidence():
        pass

    def load_nn_samples():
        pass

    def load_errors():
        pass