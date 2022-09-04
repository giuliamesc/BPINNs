import algorithms as alg
import setup.config as sc

def switch_algorithm(method):
    """ Returns an instance of the class corresponding to the selected method """
    match method:
        case "TEST": return alg.TEST
        case "HMC" : return alg.HMC
        case "SVGD": return alg.SVGD
        case "VI"  : return alg.VI
        case _ : raise Exception("This algorithm does not exist!")

def switch_problem(dataset_name):
    match dataset_name:
        case "Laplace1D_cos": return sc.Laplace1D_cos()
        case _ : raise Exception("This dataset configuration does not exist!")