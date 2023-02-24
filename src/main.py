# %% Utilities
from utility import set_config, set_directory, set_warning, starred_print
from utility import load_json, check_dataset, create_directories
from utility import switch_dataset, switch_equation, switch_configuration

# Setup utilities
set_directory()
set_warning()

# %% Import Local Classes

from setup import Parser, Param             # Setup
from setup import DataGenerator, Dataset    # Dataset Creation
from networks import BayesNN                # Models
from algorithms import Trainer              # Algorithms
from postprocessing import Storage, Plotter # Postprocessing

# %% Creating Parameters

starred_print("START")
configuration_file = switch_configuration(7, False) # Select the configuration file
args   = Parser().parse_args()   # Load a param object from command-line
config_file = set_config(args.config, configuration_file)
config = load_json(config_file)  # Load params from config file
params = Param(config, args)     # Combines args and config

data_config = switch_dataset(params.problem, params.case_name)
params.data_config = data_config

print(f"Bayesian PINN using {params.method}")
print(f"Solve the {params.inverse} problem of {params.pde} {params.phys_dim.n_input}D ")
starred_print("DONE")

# %% Datasets Creation

print("Dataset Creation")
if params.utils["gen_flag"]:
    print("\tGenerating new dataset...")
    DataGenerator(data_config) 
else:
    check_dataset(data_config)
    print(f"\tStored dataset used: {data_config.name}")

dataset = Dataset(params)
starred_print("DONE")

# %% Model Building

print("Building the Model")
print(f"\tChosing {params.pde} equation...")
equation = switch_equation(params.problem)
print("\tInitializing the Bayesian PINN...")
bayes_nn = BayesNN(params, equation) # Initialize the Bayesian NN
starred_print("DONE")

# %% Model Training

print(f"Building all algorithms...")
train_algorithm = Trainer(bayes_nn, params, dataset)
train_algorithm.pre_train()
starred_print("DONE")
train_algorithm.train()
starred_print("DONE")

# %% Model Evaluation

test_data = dataset.data_test
print("Computing solutions...")
functions_confidence = bayes_nn.mean_and_std(test_data["dom"])
functions_nn_samples = bayes_nn.draw_samples(test_data["dom"])
print("Computing errors...")
errors = bayes_nn.test_errors(functions_confidence, test_data)
print("Showing errors...")
bayes_nn.show_errors(errors)
starred_print("DONE")

# %% Saving

print("Building saving directories...")
path_folder  = create_directories(params)
save_storage = Storage(path_folder)

print("Saving data...")
# Saving Details and Results
save_storage.save_parameter(params)
save_storage.save_errors(errors)
# Saving Dataset
save_storage.data = dataset.data_plot
# Saving Training
save_storage.history  = bayes_nn.history
save_storage.thetas   = bayes_nn.thetas
# Saving Predictions
save_storage.confidence = functions_confidence
save_storage.nn_samples = functions_nn_samples
starred_print("DONE")

# %% Plotting

print("Loading data...")
plotter = Plotter(path_folder)
load_storage = Storage(path_folder)
print("Plotting the history...")
plotter.plot_losses(load_storage.history)
print("Plotting the results...")
plotter.plot_confidence(load_storage.data, load_storage.confidence)
plotter.plot_nn_samples(load_storage.data, load_storage.nn_samples)
starred_print("END")

plotter.show_plot()
