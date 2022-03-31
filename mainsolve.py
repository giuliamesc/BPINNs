# %% Import Standard Packages

import json
import os
import sys
import time
import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np

# %% Import Local Classes

sys.path.append("src")
sys.path.append("src\data_and_setup")
sys.path.append("src\models")
sys.path.append("src\postprocessing")

from args import args   #command-line arg parser
from param import param #parameter class

from helpers import create_directories

from dataset_creation import dataset_class
from dataloader import dataloader

from BayesNN import MCMC_BayesNN
# from BayesNN import SVGD_BayesNN # WORK IN PROGRESS
from auto_diff import laplace

from SVGD import SVGD
from HMC_MCMC import HMC_MCMC

from compute_error import compute_error
from plotter_old import plot_log_betas

# %% Creating Parameters

verbose = False

# Load the json file with all the parameters
with open(os.path.join("config",args.config)) as hpFile:
    hp = json.load(hpFile)

# Create a param object with hp (param from json file) and args (command-line param)
par = param(hp, args)

# Print all the selected parameters
if (verbose):
    print("--------------------------------------------")
    par.print_parameter()

# Build the directories
path_result, path_plot, path_weights = create_directories(par)
# Save parameters
par.save_parameter(path_result)


print("--------------------------------------------")
print("Bayesian PINN with", par.method)
print("Solve the inverse problem of "+str(par.n_input)+ "D " + par.pde)
print("Dataset used:", par.experiment["dataset"])

print("--------------------------------------------")
print("Dataset creation...")
# Datasets Creation
datasets_class = dataset_class(par)

# Plot the exact data
#datasets_class.plot(path_plot)

print("\tNumber of exact data:", datasets_class.n_exact)
print("\tNumber of collocation data:", datasets_class.n_collocation)
print("Building dataloader...")
# Build the dataloader for minibatch training (of just collocation points)
batch_size = par.experiment["batch_size"]
reshuffle_every_epoch = True
batch_loader  = dataloader(datasets_class, batch_size, reshuffle_every_epoch)
batch_loader, batch_loader_size = batch_loader.dataload_collocation()
print("Done")

# %% Model Building

print("--------------------------------------------")
print("Building the PDE constraint...")
# Build the pde constraint class that implements the computation of pde residual for each collocation point
if(par.pde == "laplace"):
    pinn_loss = laplace(par)
else:
    raise Exception("No other pde implemented")

print("Initializing the Bayesian PINN...")
# Initialize the correct Bayesian NN (SVGD_BayesNN for "SVGD" method, MCMC_BayesNN for every other MCMC-like method)
if(par.method == "SVGD"):
    if(par.pde == "laplace"):
        bayes_nn = SVGD_BayesNN(par.param_method["n_samples"], par.sigmas,
                                par.n_input, par.architecture,
                                par.n_out_sol, par.n_out_par, par.param,
                                pinn_loss, par.param["random_seed"])
else:
    if(par.pde == "laplace"):
        bayes_nn = MCMC_BayesNN(par.sigmas, par.n_input, par.architecture,
                                par.n_out_sol, par.n_out_par, par.param,
                                pinn_loss, par.param["random_seed"], par.param_method["M_HMC"])

print("Building", par.method ,"algorithm...")

# Build the method class
if(par.method == "SVGD"):
    # Initialize SVGD
    alg = SVGD(bayes_nn, batch_loader, datasets_class, par.param_method)
elif(par.method == "HMC"):
    # Initialize HMC
    alg = HMC_MCMC(bayes_nn, batch_loader, datasets_class,
                par.param_method, par.param["random_seed"], par.utils["debug_flag"])
else:
    raise Exception("Method not found")

print("Done")

# %% Training

print("--------------------------------------------")
print('Start training...')
t0 = time.time()
rec_log_betaD, rec_log_betaR, LOSS,LOSS1,LOSSD = alg.train_all(par.utils["verbose"])
training_time = time.time() - t0
print('End training')
print('Finished in', str(datetime.timedelta(seconds=int(training_time))))

print("--------------------------------------------")
print("Computing errors...")
# create the class to compute results and error
c_e = compute_error(par.n_out_sol, par.n_out_par, bayes_nn, datasets_class, path_result)
# compute errors and return mean and std for both outputs
functions_confidence, functions_nn_samples, errors = c_e.error()
print("Done")

# %% Saving

print("--------------------------------------------")
print("Saving networks weights...")
bayes_nn.save_networks(path_weights)

print("Save losses...")
np.savetxt(os.path.join(path_result,"Loss.csv" ),LOSS)
np.savetxt(os.path.join(path_result,"Collocation.csv"),LOSS1)
np.savetxt(os.path.join(path_result,"Fitting.csv"),LOSSD)

if (par.sigmas["data_prior_noise_trainable"] or par.sigmas["pde_prior_noise_trainable"]):
    print("Save log betass...")
    rec_log_betaD = np.array(rec_log_betaD)
    rec_log_betaR = np.array(rec_log_betaR)
    np.save(os.path.join(path_result,"log_betaD.npy"),rec_log_betaD)
    np.save(os.path.join(path_result,"log_betaR.npy"),rec_log_betaR)
print("Done")

# %% Plotting
from plotter import load_losses, plot_losses, plot_confidence, plot_nn_samples

print("--------------------------------------------")
print("Plotting the losses...")
losses = load_losses(path_result)
plot_losses(path_plot, losses)

print("Plotting the results...")
plot_confidence(path_plot, datasets_class, functions_confidence, par.n_out_sol, par.n_out_par)
plot_nn_samples(path_plot, datasets_class, functions_nn_samples, par.n_out_sol, par.n_out_par, par.method)

if (par.sigmas["data_prior_noise_trainable"] or par.sigmas["pde_prior_noise_trainable"]):
    print("Plotting log betas")
    plot_log_betas(rec_log_betaD, rec_log_betaR, path_plot)

print("End")
print("--------------------------------------------")