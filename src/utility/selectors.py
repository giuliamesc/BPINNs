import algorithms as alg
import datasets   as data
import equations  as eq

def switch_algorithm(method):
    """ Returns an instance of the class corresponding to the selected method """
    match method:
        case "TEST": return alg.TEST
        case "HMC" : return alg.HMC
        case "SVGD": return alg.SVGD
        case "VI"  : return alg.VI
        case _ : raise Exception("This algorithm does not exist!")

def switch_dataset(dataset_name):
    match dataset_name:
        case "Laplace1D_cos": return data.Laplace1D_cos()
        case _ : raise Exception("This dataset configuration does not exist!")

def switch_equation(dataset_name):
    match dataset_name:
        case "Laplace1D_cos": return eq.Laplace
        case "Laplace1D_cos": return eq.Laplace
        case _ : raise Exception("This equation does not exist!")
