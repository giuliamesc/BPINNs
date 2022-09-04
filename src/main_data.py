from utility import set_directory, set_gui_len
from utility import switch_problem
from setup import AnalyticalData

set_directory()
gui_len = set_gui_len()

AnalyticalData(switch_problem("Laplace1D_cos"), gui_len, do_plots=True, test_only=False, save_plot=True, is_main=True).show_plot()
#AnalyticalData(switch_problem("Laplace2D_cos"), gui_len, do_plots=True, test_only=True, save_plot=True, is_main=True).show_plot()
