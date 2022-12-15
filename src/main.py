# %% Utilities
from utility import set_config, set_directory, set_warning, starred_print
from utility import load_json, check_dataset, create_directories
from utility import switch_algorithm, switch_dataset, switch_equation

# Manual configuration
test_cases = [None, "ADAM_regression", "ADAM_laplace", "HMC_regression", "HMC_laplace", "SVGD_regression"]
best_cases = [None, "ADAM_lap_cos", "HMC_lap_cos", "HMC_reg_cos", "HMC_reg_sin"]
test_mode, case_num = True, 1 # True for test_cases, False for best_cases
configuration_file = "test_models/" + test_cases[case_num] if test_mode else "best_models/" + best_cases[case_num]

# Setup utilities
set_directory()
set_warning()

# %% Import Local Classes

from setup import Parser, Param             # Setup
from setup import DataGenerator, Dataset    # Dataset Creation
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
