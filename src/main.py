# %% Utilities
from utility import set_config, set_directory, set_warning, set_gui_len
from utility import load_json, check_dataset, create_directories
from utility import switch_algorithm, switch_dataset, switch_equation

# Setup utilities
configuration_file = None
set_directory()
set_warning()
gui_len = set_gui_len()

# %% Import Local Classes

from setup import Parser, Param             # Setup
from setup import AnalyticalData            # Data Generation
from setup import Dataset, Dataloader       # Dataset Creation
from networks import BayesNN                # Models
from postprocessing import Storage, Plotter # Postprocessing

# %% Creating Parameters

print(" START ".center(gui_len,'*'))
args   = Parser().parse_args()   # Load a param object from command-line
config_file = set_config(args.config, configuration_file)
config = load_json(config_file)  # Load params from config file
params = Param(config, args)     # Combines args and config

data_config = switch_dataset(params.problem, params.case_name)
params.data_config = data_config
debug_flag  = params.utils["debug_flag"]

print("Bayesian PINN with", params.method)
print("Solve the inverse problem of " + str(params.phys_dim.n_input) + "D " + params.pde)
print(" DONE ".center(gui_len,'*'))


# %% Datasets Creation

print("Dataset Creation")
if params.utils["gen_flag"]:
    print("\tGenerating new dataset...")
    AnalyticalData(data_config, gui_len)
else:
    check_dataset(data_config)
    print("\tStored dataset used:", data_config.name)

dataset = Dataset(params)
print("\tNumber of fitting data:", dataset.num_fitting)
print("\tNumber of collocation data:", dataset.num_collocation)
print(" DONE ".center(gui_len,'*'))

#print("Building dataloader...") 
#batch_loader = Dataloader(dataset, params.experiment["batch_size"], params.utils['random_seed'])
#batch_loader = batch_loader.dataload_collocation() # Build the dataloader for minibatch training
#print(" DONE ".center(gui_len,'*'))

# %% Model Building

print("Building the Model")
print("\tChosing", params.pde, "equation...")
equation = switch_equation(params.problem)
print("\tInitializing the Bayesian PINN...")
bayes_nn = BayesNN(params, equation) # Initialize the Bayesian NN
print("\tChosing", params.method, "algorithm...")
chosen_algorithm = switch_algorithm(params.method) # Chose the algorithm from config/args
print("\tBuilding", params.method, "algorithm...")
train_algorithm = chosen_algorithm(bayes_nn, params.param_method, debug_flag) # Initialize the algorithm chosen
train_algorithm.data_train = dataset # Insert the dataset used for training # Decidi se separare qua in batch
print(" DONE ".center(gui_len,'*'))

# %% Training

print('Start training...')
train_algorithm.train() # Create list of theta samples
print(" DONE ".center(gui_len,'*'))

# %% Model Evaluation

print("Computing solutions...")
functions_confidence = bayes_nn.mean_and_std(dataset.dom_data[0])
functions_nn_samples = bayes_nn.draw_samples(dataset.dom_data[0])
print("Computing errors...")
errors = bayes_nn.test_errors(functions_confidence, dataset)
print("Showing errors...")
bayes_nn.show_errors(errors)
print(" DONE ".center(gui_len,'*'))

# %% Saving

print("Building saving directories...")
path_plot, path_data, path_values, path_thetas, path_log = create_directories(params)
save_storage = Storage(path_data, path_values, path_thetas, path_log)

print("Saving data...")
# Saving Details and Results
save_storage.save_parameter(params)
save_storage.save_errors(errors)
# Saving Training
save_storage.history  = bayes_nn.history
save_storage.thetas   = bayes_nn.thetas
# Saving Sigmas
save_storage.sg_flags = bayes_nn.sg_flags
save_storage.sigmas   = bayes_nn.sigmas
# Saving Predictions
save_storage.confidence = functions_confidence
save_storage.nn_samples = functions_nn_samples
print(" DONE ".center(gui_len,'*'))

# %% Plotting

print("Loading data...")
plotter = Plotter(path_plot)
load_storage = Storage(path_data, path_values, path_thetas, path_log)
print("Plotting the history...")
history = load_storage.history
plotter.plot_losses(history)
sigmas  = load_storage.sigmas
plotter.plot_sigmas(sigmas)
print("Plotting the results...")
functions_confidence = load_storage.confidence
functions_nn_samples = load_storage.nn_samples
plotter.plot_confidence(dataset.dom_data, dataset.exact_data_noise, functions_confidence)
plotter.plot_nn_samples(dataset.dom_data, dataset.exact_data_noise, functions_nn_samples)
print(" END ".center(gui_len,'*'))

#plotter.show_plot()