from utility import set_directory, set_gui_len
from utility import create_paths
from postprocessing import Storage, Plotter
import os

set_directory()
gui_len = set_gui_len()

prob_name = "Laplace1D"
prob_path = os.path.join("../outs", prob_name)

case_name = "default"
case_path = os.path.join(prob_path, case_name)

test_name = "trash"
test_path = os.path.join(case_path, test_name)

path_plot, path_data, path_values, path_thetas, path_log = create_paths(test_path)
print("Loading data...")
plotter = Plotter(path_plot)
load_storage = Storage(path_data, path_values, path_thetas, path_log)
loaded_data  = load_storage.data
print("Plotting the history...")
history = load_storage.history
plotter.plot_losses(history)
sigmas  = load_storage.sigmas
plotter.plot_sigmas(sigmas)
print("Plotting the results...")
functions_confidence = load_storage.confidence
functions_nn_samples = load_storage.nn_samples
plotter.plot_confidence(loaded_data[0], loaded_data[1], functions_confidence)
plotter.plot_nn_samples(loaded_data[0], loaded_data[1], functions_nn_samples)
print(" END ".center(gui_len,'*'))

plotter.show_plot()