from .config import Laplace1D_cos

def switch_problem(dataset_name):
    match dataset_name:
        case "Laplace1D_cos": return Laplace1D_cos()
        case _ : raise Exception("This dataset configuration does not exist!")