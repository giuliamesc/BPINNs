from utility import set_directory, set_gui_len
from utility import switch_dataset
from setup import AnalyticalData

set_directory()
gui_len = set_gui_len()

AnalyticalData(switch_dataset("Laplace1D","cos"), gui_len, 
               do_plots=True, test_only=True, is_main=True).show_plot()
