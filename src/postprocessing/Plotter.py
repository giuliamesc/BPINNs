import matplotlib.pyplot as plt
import numpy as np
import os

class Plotter():
    """ 
    Class for plotting utilities:
    Methods:
        - plot_losses: plots MSE and log-likelihood History
        - plot_nn_samples: plots all samples of solution and parametric field
        - plot_confidence: plots mean and std of solution and parametric field
        - show_plot: enables plot visualization
    """
    
    def __init__(self, path_folder):
        
        self.path_plot = os.path.join(path_folder,"plot")
        self.path_log  = os.path.join(path_folder,"log")
        with open(os.path.join(self.path_log,"parameters.txt"), "r") as file_params:
            problem = file_params.readlines()[2][10:].strip()
        self.only_sol = problem == "Regression"

    def __order_inputs(self, inputs):
        """ Sorting the input points by label """
        idx = np.argsort(inputs)
        inputs = inputs[idx]
        return inputs, idx

    def __save_plot(self, path, title):
        """ Auxiliary function used in all plot functions for saving """
        path = os.path.join(path, title)
        plt.savefig(path, bbox_inches = 'tight')

    def __plot_confidence_1D(self, x, func, title, label = ("",""), fit = None):
        """ Plots mean and standard deviation of func (1D case); used in plot_confidence """
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

    def __plot_nn_samples_1D(self, x, func, label = ("",""), fit = None):
        """ Plots all the samples of func; used in plot_nn_samples """
        x, idx = self.__order_inputs(x)

        plt.figure()
        blurring = min(1.0, 2/len(func[1]))
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

    def __plot_train(self, losses, name, title):
        """ Plots all the loss history; used in plot_losses """
        plt.figure()
        x = list(range(1,len(losses['Total'])+1))
        if name[:-4] == "LogLoss":
            plt.semilogx(x, losses['Total'], 'k--', lw=2.0, alpha=1.0, label = 'Total')
        for key, value in losses.items():
            if key == "Total": continue
            plt.semilogx(x, value, lw=1.0, alpha=0.7, label = key)
        plt.title(f"History of {title}")
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.legend(prop={'size': 9})
        self.__save_plot(self.path_plot, title)

    def plot_confidence(self, dom_data, fit_data, functions):
        """ Plots mean and standard deviation of solution and parametric field samples """
        inputs, u_true, f_true = dom_data
        ex_points, u_values, f_values = fit_data

        u = (u_true, functions['sol_NN'], functions['sol_std'])
        u_fit = (ex_points, u_values)
        f = (f_true, functions['par_NN'], functions['par_std'])
        f_fit = (ex_points, f_values)

        self.__plot_confidence_1D(inputs[:,0], u, 'Confidence interval for u(x)', label = ('x','u'), fit = u_fit)
        self.__save_plot(self.path_plot, 'u_confidence.png')
        if self.only_sol: return
        self.__plot_confidence_1D(inputs[:,0], f, 'Confidence interval for f(x)', label = ('x','f'), fit = f_fit)
        self.__save_plot(self.path_plot, 'f_confidence.png')

    def plot_nn_samples(self, dom_data, fit_data, functions):
        """ Plots all the samples of solution and parametric field """
        inputs, u_true, f_true = dom_data
        ex_points, u_values, f_values = fit_data

        u = (u_true, functions['sol_samples'])
        u_fit = (ex_points, u_values)
        f = (f_true, functions['par_samples'])
        f_fit = (ex_points, f_values)

        self.__plot_nn_samples_1D(inputs[:,0], u, label = ('x','u'), fit = u_fit)
        self.__save_plot(self.path_plot, 'u_nn_samples.png')
        if self.only_sol: return
        self.__plot_nn_samples_1D(inputs[:,0], f, label = ('x','f'), fit = f_fit)
        self.__save_plot(self.path_plot, 'f_nn_samples.png')

    def plot_losses(self, losses):
        """ Generates the plots of MSE and log-likelihood """
        self.__plot_train(losses[0], "Loss.png"   , "Mean Squared Error")
        self.__plot_train(losses[1], "LogLoss.png", "Loss (Log-Likelihood)")

    def __wait_input(self, key):
        """ Start a loop that will run until the user enters key """
        key_input = ''
        while key_input != key:
            key_input = input("Input Q to quit: ").upper()

    def show_plot(self):
        """ Shows the plots """
        plt.show(block = False)
        self.__wait_input('Q')
