from utility import set_directory
from utility import switch_dataset
from setup import AnalyticalData

set_directory()
AnalyticalData(switch_dataset("Laplace1D","cos"), 
               do_plots=True, test_only=True, is_main=True).show_plot()
