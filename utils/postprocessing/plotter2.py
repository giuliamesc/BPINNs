import matplotlib.pyplot as plt
import numpy as np
import os
import csv

def plot_losses(path_plot, losses):
    """
    Plot log(losses)
    """
    plt.figure()
    plt.plot(np.log(losses['Loss']), 'k--', lw=2.5, alpha=1.0, label = 'Loss_Total')
    for key,value in losses.items():
        if not key == 'Loss':
            plt.plot(np.log(value), lw=1.0, alpha=0.7, label = key)
    plt.xlabel('epochs')
    plt.ylabel('LogLoss')
    plt.legend(prop={'size': 9})
    path = os.path.join(path_plot,"loss.png")
    plt.savefig(path, bbox_inches= 'tight')
    
def load_losses(path_result):
    losses = dict()
    for loss_filename in os.listdir(path_result):
        if loss_filename[-4:] == ".csv":
            with open(os.path.join(path_result,loss_filename)) as loss_file:
                csvreader = csv.reader(loss_file)
                loss_list = list()
                for loss_value in csvreader:
                    loss_list.append(loss_value)
            losses[loss_filename[:-4]] = np.array(loss_list, dtype='float32')
    return losses

def plot_result2(path_plot, datasets_class, functions, n_out_sol, n_out_par):
    inputs, u_true, f_true = datasets_class.get_dom_data()
    u_points, u_values, _  = datasets_class.get_exact_data_with_noise()
    
    u = (u_true, functions['u_NN'], functions['u_std'])
    u_fit = (u_points, u_values)
    f = (f_true, functions['f_NN'], functions['f_std'])
    
    plot_1D(inputs, u, 'Confidence interval for u(x)', label = ('x','u'), fit = u_fit)
    save_plot(path_plot, 'u.png')
    plot_1D(inputs, f, 'Confidence interval for f(x)', label = ('x','f'))
    save_plot(path_plot, 'f.png')
    

def plot_1D(x, func, title, label = ("",""), fit = None):
    plt.figure()
    plt.plot(x, func[0], 'r-', label='true')
    plt.plot(x, func[1], 'b--', label='mean')
    plt.plot(x, func[1] - func[2], 'g--', label='mean-std')
    plt.plot(x, func[1] + func[2], 'g--', label='mean+std')
    if fit is not None:
        plt.plot(fit[0], fit[1], 'r*')

    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.legend(prop={'size': 9})
    plt.title(title)

    
def save_plot(path_plot, title):
    path = os.path.join(path_plot, title)
    plt.savefig(path,bbox_inches= 'tight')

if __name__ == "__main__":
    my_path = '../../1D-laplace/results/trash'
    losses = load_losses(my_path)
    plot_losses(my_path, **losses)
    
