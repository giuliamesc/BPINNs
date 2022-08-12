# %% Import Standard Packages

import json
import os
import logging

# Move into src if necessary
if os.getcwd()[-3:] != "src":
    new_dir = os.path.join(os.getcwd(),"src")
    os.chdir(new_dir)
    print(f"Working Directory moved to: {new_dir}")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logging.basicConfig(level=logging.ERROR)
gui_len = max(50,int(os.get_terminal_size().columns/3))
# %% Import Local Classes

# Setup
from data_and_setup.args import Parser #command-line arg parser
from data_and_setup.param import Param #parameter class
from data_and_setup.create_directories import create_directories
# Dataset Creation
from data_and_setup.dataset_creation import dataset_class
from data_and_setup.dataloader import dataloader
# Model
from networks.BayesNN import BayesNN
# Algorithms
from algorithms.HMC import HMC
# Postprocessing
from post_processing import Storage
from post_processing import Plotter

#from __postprocessing.plotter import load_losses, plot_losses, plot_confidence, plot_nn_samples, show_plot

# %% Creating Parameters

# Load a param object from command-line
args = Parser().parse_args()

# Load the json file with all the parameters
with open(os.path.join("../config",args.config+".json")) as hpFile:
    hp = json.load(hpFile)

# Combine a param object with hp (param from json file) and args (command-line param)
par = Param(hp, args)

# Build the directories
path_result, path_plot, path_weights = create_directories(par)
# Save parameters
par.save_parameter(path_result)

print(" START ".center(gui_len,'*'))
print("Bayesian PINN with", par.method)
print("Solve the inverse problem of " + str(par.n_input) + "D " + par.pde)
print("Dataset used:", par.experiment["dataset"])

print(" DONE ".center(gui_len,'*'))

# %% Datasets Creation
print("Dataset creation...")
datasets_class = dataset_class(par)

print("\tNumber of fitting data:", datasets_class.num_fitting)
print("\tNumber of collocation data:", datasets_class.num_collocation)
print("Building dataloader...")
# Build the dataloader for minibatch training (of just collocation points)

batch_loader = dataloader(datasets_class, par.experiment["batch_size"], par.utils['random_seed'])
batch_loader = batch_loader.dataload_collocation()
print(" DONE ".center(gui_len,'*'))

# %% Model Building

print("Initializing the Bayesian PINN...")
# Initialize the correct Bayesian NN
bayes_nn = BayesNN(par)

print("Chosing", par.method ,"algorithm...")
chosen_algorithm = HMC(bayes_nn)
""" Switch tra gli algoritmi """

print("Building", par.method ,"algorithm...")
# Initialize the algorithm chosen
train_algorithm = chosen_algorithm
# Insert the dataset used for training
train_algorithm.data_train = datasets_class # Decidi se separare qua in batch

print(" DONE ".center(gui_len,'*'))

# %% Training

print('Start training...')
# Create list of theta samples
train_algorithm.train(par)

print('End training')
train_algorithm.compute_time()
print(" DONE ".center(gui_len,'*'))

# %% Model Evaluation

print("Computing solutions...")
functions_confidence = bayes_nn.mean_and_std()
functions_nn_samples = bayes_nn.draw_samples()

print("Computing errors...")
errors = bayes_nn.compute_errors()
""" STAMPA ERRORI """

print(" DONE ".center(gui_len,'*'))

# %% Saving

save_storage = Storage(path_result, path_plot, path_weights)
save_storage.save_training(bayes_nn.theta, train_algorithm.loss)
save_storage.save_results(functions_confidence, functions_nn_samples)
save_storage.save_errors(errors)

# %% Plotting

print("Loading data")
plotter = Plotter()
load_storage = Storage(path_result, path_plot, path_weights)

print("Plotting the losses...")
losses = load_storage.load_losses()
plotter.plot_losses(path_plot, losses)

print("Plotting the results...")
functions_confidence = load_storage.load_confidence()
functions_nn_samples = load_storage.load_nn_samples()
plotter.plot_confidence(path_plot, datasets_class, functions_confidence, par.n_out_sol, par.n_out_par)
plotter.plot_nn_samples(path_plot, datasets_class, functions_nn_samples, par.n_out_sol, par.n_out_par, par.method)

print(" END ".center(gui_len,'*'))
plotter.show_plot()
