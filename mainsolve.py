# %% Import Standard Packages

import json
import os
import sys
import time
import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np

# %% Import Local Classes

sys.path.append("utils")
sys.path.append("utils\data_and_setup")
sys.path.append("utils\models")
sys.path.append("utils\postprocessing")

from args import args   #command-line arg parser
from param import param #parameter class

from helpers import create_directories

from dataset_creation import dataset_class
from dataloader import dataloader

from BayesNN import SVGD_BayesNN
from BayesNN import MCMC_BayesNN
from pde_constraint import laplace

from SVGD import SVGD
from HMC_MCMC import HMC_MCMC

from compute_error import compute_error
from plotter import plot_result, plot_losses, plot_log_betas, plot_all_result

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
    pde_constr = laplace(par)
else:
    raise Exception("No other pde implemented")

print("Initializing the Bayesian PINN...")
# Initialize the correct Bayesian NN (SVGD_BayesNN for "SVGD" method, MCMC_BayesNN for every other MCMC-like method)
if(par.method == "SVGD"):
    if(par.pde == "laplace"):
        bayes_nn = SVGD_BayesNN(par.param_method["n_samples"], par.sigmas,
                                par.n_input, par.architecture,
                                par.n_out_sol, par.n_out_par, par.param,
                                pde_constr, par.param["random_seed"])
else:
    if(par.pde == "laplace"):
        bayes_nn = MCMC_BayesNN(par.sigmas, par.n_input, par.architecture,
                                par.n_out_sol, par.n_out_par, par.param,
                                pde_constr, par.param["random_seed"], par.param_method["M_HMC"])

print("Building", par.method ,"algorithm...")

# Build the method class
if(par.method == "SVGD"):
    # Initialize SVGD
    alg = SVGD(bayes_nn, batch_loader, datasets_class, par.param_method)
elif(par.method == "HMC"):
    # Initialize HMC
    alg = HMC_MCMC(bayes_nn, batch_loader, datasets_class,
                par.param_method, par.param["random_seed"])
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
u_NN, f_NN, u_std, f_std, errors = c_e.error()
print("Done")

# %% Saving

print("--------------------------------------------")
print("Saving networks weights...")
bayes_nn.save_networks(path_weights)

print("Save losses...")
np.savetxt(os.path.join(path_result,"Loss.csv" ),LOSS)
np.savetxt(os.path.join(path_result,"LOSS1.csv"),LOSS1)
np.savetxt(os.path.join(path_result,"LOSSD.csv"),LOSSD)

if (par.sigmas["data_prior_noise_trainable"] or par.sigmas["pde_prior_noise_trainable"]):
    print("Save log betass...")
    rec_log_betaD = np.array(rec_log_betaD)
    rec_log_betaR = np.array(rec_log_betaR)
    np.save(os.path.join(path_result,"log_betaD.npy"),rec_log_betaD)
    np.save(os.path.join(path_result,"log_betaR.npy"),rec_log_betaR)
print("Done")

# # %% Plotting

# print("--------------------------------------------")
# print("Plotting the losses...")
# plot_losses(LOSSD, LOSS1, LOSS, path_plot)
# print("Plotting the results...")
# plot_result(par.n_out_par, u_NN, f_NN, u_std, f_std, datasets_class, path_plot)

# print("Plot all the NNs...")
# if(par.n_input == 1):
#     inputs, u, f = datasets_class.get_dom_data()
#     u_NN, f_NN = bayes_nn.predict(inputs)
#     x = inputs[:,0]
#     plot_all_result(x, u, f, u_NN, f_NN, datasets_class,
#                     par.n_input, par.n_output_vel, par.method, path_plot)
# else:
#     print("Unable to plot all the NNs in 1D up to now")
# if (par.sigmas["data_prior_noise_trainable"] or par.sigmas["pde_prior_noise_trainable"]):
#     print("Plotting log betas")
#     plot_log_betas(rec_log_betaD, rec_log_betaR, path_plot)

# print("End")
# print("--------------------------------------------")
