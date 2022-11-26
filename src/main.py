# %% Utilities
from utility import set_config, set_directory, set_warning, starred_print
from utility import load_json, check_dataset, create_directories
from utility import switch_algorithm, switch_dataset, switch_equation

# Manual configuration
test_cases = [None, "ADAM_regression", "HMC_regression", "HMC_laplace"]
configuration_file = test_cases[1]

# Setup utilities
set_directory()
set_warning()

# %% Import Local Classes

from setup import Parser, Param             # Setup
from setup import AnalyticalData, Dataset   # Dataset Creation
from networks import BayesNN                # Models
from postprocessing import Storage, Plotter # Postprocessing

# %% Creating Parameters

starred_print("START")
args   = Parser().parse_args()   # Load a param object from command-line
config_file = set_config(args.config, configuration_file)
config = load_json(config_file)  # Load params from config file
params = Param(config, args)     # Combines args and config

data_config = switch_dataset(params.problem, params.case_name)
params.data_config = data_config
debug_flag  = params.utils["debug_flag"]

print(f"Bayesian PINN using {params.method}")
print(f"Solve the {params.inverse} problem of {params.pde} {params.phys_dim.n_input}D ")
starred_print("DONE")


# %% Datasets Creation

print("Dataset Creation")
if params.utils["gen_flag"]:
    print("\tGenerating new dataset...")
    AnalyticalData(data_config)
else:
    check_dataset(data_config)
    print(f"\tStored dataset used: {data_config.name}")

dataset = Dataset(params)
print(f"\tNumber of fitting data: {dataset.num_fitting}")
print(f"\tNumber of collocation data: {dataset.num_collocation}")
starred_print("DONE")

# %% Model Building

print("Building the Model")
print(f"\tChosing {params.pde} equation...")
equation = switch_equation(params.problem)
print("\tInitializing the Bayesian PINN...")
bayes_nn = BayesNN(params, equation) # Initialize the Bayesian NN
print(f"\tChosing {params.method} algorithm...")
chosen_algorithm = switch_algorithm(params.method) # Chose the algorithm from config/args
print(f"\tBuilding {params.method} algorithm...")
train_algorithm = chosen_algorithm(bayes_nn, params.param_method, debug_flag) # Initialize the algorithm chosen
train_algorithm.data_train = dataset # Insert the dataset used for training
starred_print("DONE")

# %% Training

print('Start training...')
train_algorithm.train() # Create list of theta samples
starred_print("DONE")

# %% Model Evaluation

print("Computing solutions...")
functions_confidence = bayes_nn.mean_and_std(dataset.dom_data[0])
functions_nn_samples = bayes_nn.draw_samples(dataset.dom_data[0])
print("Computing errors...")
errors = bayes_nn.test_errors(functions_confidence, dataset)
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
save_storage.data = (dataset.dom_data, dataset.noise_data) 
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
loaded_data  = load_storage.data
plotter.plot_confidence(loaded_data[0], loaded_data[1], load_storage.confidence)
plotter.plot_nn_samples(loaded_data[0], loaded_data[1], load_storage.nn_samples)
starred_print("END")

plotter.show_plot()