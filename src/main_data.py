from utility import set_directory
from utility import switch_dataset
from setup import DataGenerator

set_directory()
DataGenerator(switch_dataset("Regression","cos"), False)
