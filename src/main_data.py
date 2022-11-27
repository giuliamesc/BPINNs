from utility import set_directory
from utility import switch_dataset
from setup import AnalyticalData, AnalyticalDataNew

set_directory()
#AnalyticalData(switch_dataset("Regression","cos"), do_plots=True, test_only=True, is_main=True).show_plot()
AnalyticalDataNew(switch_dataset("Regression","sin"))
