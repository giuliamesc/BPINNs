# %% Import Standard Packages

import json
import os
import time
import datetime
import numpy as np
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
""" IMPORTA EQUAZIONI """
from networks.BayesPiNN import BayesPiNN
""" IMPORTA ALGORITMO BACKPROP """

# Postprocessing
# from postprocessing.compute_error import compute_error
from postprocessing.plotter import load_losses, plot_losses, plot_confidence, plot_nn_samples, show_plot

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
print("Dataset creation...")
# Datasets Creation
datasets_class = dataset_class(par)

print("\tNumber of fitting data:", datasets_class.num_fitting)
print("\tNumber of collocation data:", datasets_class.num_collocation)
print("Building dataloader...")
# Build the dataloader for minibatch training (of just collocation points)

batch_loader = dataloader(datasets_class, par.experiment["batch_size"], par.utils['random_seed'])
batch_loader = batch_loader.dataload_collocation()
print(" DONE ".center(gui_len,'*'))

# %% Model Building

print("Building the PDE constraint...")
# Build the pde constraint class that implements the computation of pde residual for each collocation point
""" INIZIALIZZA IL PROBLEMA """

print("Initializing the Bayesian PINN...")
# Initialize the correct Bayesian NN
""" INIZIALIZZA LA RETE """
bayes_nn = BayesPiNN(par, )

print("Building", par.method ,"algorithm...")
# Build the method class
""" INIZIALIZZA L'ALGORITMO """
# eg: alg = HMC(bayes_nn, dataset)
print(" DONE ".center(gui_len,'*'))

# %% Training

print('Start training...')
t0 = time.time()
""" TRAINING """
# eg: alg.train(train par)
training_time = time.time() - t0
print('End training')
print('Finished in', str(datetime.timedelta(seconds=int(training_time))))
print(" DONE ".center(gui_len,'*'))

print("Computing errors...")
# eg: out = bayes_nn.predict()
# eg: err = bayes_nn.comperr(out)
functions_confidence, functions_nn_samples = None, None

print(" DONE ".center(gui_len,'*'))


# %% Plotting

print("Plotting the losses...")
losses = load_losses(path_result)
plot_losses(path_plot, losses)

print("Plotting the results...")
plot_confidence(path_plot, datasets_class, functions_confidence, par.n_out_sol, par.n_out_par)
plot_nn_samples(path_plot, datasets_class, functions_nn_samples, par.n_out_sol, par.n_out_par, par.method)

print(" END ".center(gui_len,'*'))
show_plot()
