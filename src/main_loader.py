from utility import set_directory, starred_print
from postprocessing import Storage, Plotter
import os

set_directory()

def print_selection(base_folder, message):
    starred_print("SELECTION")
    print(f"Available {message}s:")
    for code, folder in enumerate(os.listdir(base_folder)[1:]):
        print(f"{code+1:2d}) {folder}")

def choose_folder(base_folder, message):
    print_selection(base_folder, message)
    code = int(input(f"Select code of chosen {message}: "))
    return os.listdir(base_folder)[code]

base_path = "../outs"
prob_name = choose_folder(base_path, "problem")
prob_path = os.path.join("../outs", prob_name)
case_name = choose_folder(prob_path, "setup case")
case_path = os.path.join(prob_path, case_name)
test_name = choose_folder(case_path, "test case")
test_path = os.path.join(case_path, test_name)

starred_print("START")
print("Loading data...")
plotter = Plotter(test_path)
load_storage = Storage(test_path)
print("Plotting the history...")
plotter.plot_losses(load_storage.history)
print("Plotting the results...")
loaded_data  = load_storage.data
plotter.plot_confidence(load_storage.data, load_storage.confidence)
plotter.plot_nn_samples(load_storage.data, load_storage.nn_samples)
starred_print("END")

plotter.show_plot()