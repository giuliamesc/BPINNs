# %% Utilities
from utility import set_directory, set_warning, set_gui_len
from utility import load_json, create_directories
from utility import switch_algorithm

# Setup utilities
set_directory()
set_warning()
gui_len = set_gui_len()

# %% Import Local Classes

# Setup
from setup import Parser, Param
# Dataset Creation
from setup import Dataset, Dataloader
# Model
from networks import BayesNN
# Postprocessing
from postprocessing import Storage, Plotter

# %% Creating Parameters

# Load a param object from command-line
args = Parser().parse_args()
# Load params from config file
hp = load_json(args.config)
# Combine a param object with hp (param from json file) and args (command-line param)
par = Param(hp, args)

print(" START ".center(gui_len,'*'))
print("Bayesian PINN with", par.method)
print("Solve the inverse problem of " + str(par.phys_dim.n_input) + "D " + par.pde)
print("Dataset used:", par.experiment["dataset"])

print(" DONE ".center(gui_len,'*'))

# %% Datasets Creation
print("Dataset creation...")
dataset = Dataset(par)
print("\tNumber of fitting data:", dataset.num_fitting)
print("\tNumber of collocation data:", dataset.num_collocation)

print("Building dataloader...")
# Build the dataloader for minibatch training (of just collocation points)
batch_loader = Dataloader(dataset, par.experiment["batch_size"], par.utils['random_seed'])
batch_loader = batch_loader.dataload_collocation()
print(" DONE ".center(gui_len,'*'))

# %% Model Building

print("Initializing the Bayesian PINN...")
# Initialize the correct Bayesian NN
bayes_nn = BayesNN(par)

print("Chosing", par.method ,"algorithm...")
# Chose the algorithm from config/args
chosen_algorithm = switch_algorithm(par, test = True)

print("Building", par.method ,"algorithm...")
# Initialize the algorithm chosen
train_algorithm = chosen_algorithm(bayes_nn)
# Insert the dataset used for training
train_algorithm.data_train = dataset # Decidi se separare qua in batch

print(" DONE ".center(gui_len,'*'))

# %% Training

print('Start training...')
# Create list of theta samples
train_algorithm.train(par)

print('End training')
# Compute duration of training
train_algorithm.compute_time()

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
path_plot, path_values, path_weights, path_log = create_directories(par)
save_storage = Storage(path_values, path_weights, path_log)

print("Saving data...")
#save_storage.save_parameter(par) (in txt)
#save_storage.save_training(bayes_nn.thetas, train_algorithm.loss)
save_storage.confidence = functions_confidence
save_storage.nn_samples = functions_nn_samples
#save_storage.save_errors(errors) (in txt)

print(" DONE ".center(gui_len,'*'))

# %% Plotting

print("Loading data...")
plotter = Plotter(path_plot)
load_storage = Storage(path_values, path_weights, None)

print("Plotting the losses...")
#losses = load_storage.load_losses()
#plotter.plot_losses(losses)

print("Plotting the results...")
functions_confidence = load_storage.confidence
functions_nn_samples = load_storage.nn_samples
plotter.plot_confidence(dataset, functions_confidence)
plotter.plot_nn_samples(dataset, functions_nn_samples)

print(" END ".center(gui_len,'*'))

plotter.show_plot()
