import matplotlib.pyplot as plt
import numpy as np
import os

class Plotter():
    
    def __init__(self, path_plot):
        
        self.path_plot = path_plot

    def __save_plot(self, path, title):
        path = os.path.join(path, title)
        plt.savefig(path, bbox_inches = 'tight')

    def plot_losses(self, losses):
        pass

    def __order_inputs(self, inputs):

        idx = np.argsort(inputs)
        inputs = inputs[idx]

        return inputs, idx

    def __plot_confidence_1D(self, x, func, title, label = ("",""), fit = None):
    
        x, idx = self.__order_inputs(x)
        func = [f[idx] for f in func]

        plt.figure()
        plt.plot(x, func[0], 'r-',  label='true')
        plt.plot(x, func[1], 'b--', label='mean')
        plt.plot(x, func[1] - func[2], 'g--', label='mean-std')
        plt.plot(x, func[1] + func[2], 'g--', label='mean+std')
        if fit is not None:
            plt.plot(fit[0], fit[1], 'r*')

        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.legend(prop={'size': 9})
        plt.title(title)

    def plot_confidence(self, dataset, functions):
        
        inputs, u_true, f_true = dataset.dom_data
        u_points, u_values, _  = dataset.exact_data_noise

        u = (u_true, functions['sol_NN'], functions['sol_std'])
        u_fit = (u_points, u_values)
        f = (f_true, functions['par_NN'], functions['par_std'])

        self.__plot_confidence_1D(inputs[:,0], u, 'Confidence interval for u(x)', label = ('x','u'), fit = u_fit)
        self.__save_plot(self.path_plot, 'u_confidence.png')
        self.__plot_confidence_1D(inputs[:,0], f, 'Confidence interval for f(x)', label = ('x','f'))
        self.__save_plot(self.path_plot, 'f_confidence.png')

    def __plot_nn_samples_1D(self, x, func, label = ("",""), fit = None):

        x, idx = self.__order_inputs(x)

        plt.figure()
        blurring = 0.2/len(func[1])
        for func_sample in func[1]:
            plt.plot(x, func_sample[idx,0], 'b-', markersize=0.01, alpha=blurring)

        func_ex = func[0][idx]
        plt.plot(x, func_ex, 'r-', label='true')
        if fit is not None:
            plt.plot(fit[0], fit[1], 'r*')

        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.legend(prop={'size': 9})
        plt.title('Samples from ' + label[1] + ' reconstructed distribution')

    def plot_nn_samples(self, dataset, functions):

        inputs, u_true, f_true = dataset.dom_data
        u_points, u_values, _  = dataset.exact_data_noise

        u = (u_true, functions['sol_samples'])
        u_fit = (u_points, u_values)
        f = (f_true, functions['par_samples'])

        self.__plot_nn_samples_1D(inputs[:,0], u, label = ('x','u'), fit = u_fit)
        self.__save_plot(self.path_plot, 'u_nn_samples.png')
        self.__plot_nn_samples_1D(inputs[:,0], f, label = ('x','f'), fit = None)
        self.__save_plot(self.path_plot, 'f_nn_samples.png')

    def show_plot(self):
        
        plt.show(block = True)
        