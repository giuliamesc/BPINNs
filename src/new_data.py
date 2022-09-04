from utility import set_directory
from setup import AnalyticalData

set_directory()
AnalyticalData("Laplace1D_cos", do_plots = True, test_only = False, save_plot = True, is_main = True).show_plot()
#AnalyticalData("Laplace2D_cos", do_plots = True, test_only = True, save_plot = True, is_main = True).show_plot()
