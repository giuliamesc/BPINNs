import matplotlib.pyplot as plt

class Plotter():
    
    def __init__(self, path_plot, par):
        self.path_plot = path_plot
        self.par = par
        pass

    def plot_losses(self, losses):
        pass

    def plot_confidence(self, datasets_class, functions_confidence):
        pass

    def plot_nn_samples(self, datasets_class, functions_nn_samples):
        pass

    def show_plot(self):
        plt.show(block = True)
        