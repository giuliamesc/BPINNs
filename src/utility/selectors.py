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

def switch_dataset(problem, case_name):
    match problem:
        case "Laplace1D": 
            match case_name:
                case "default": return data.Laplace1D_default()
                case "cos":     return data.Laplace1D_cos()
                case "sin":     return data.Laplace1D_sin()
                case _ :  raise Exception("This case test does not exist!")
        case "Laplace2D": raise Exception("Not implemeted yet")
        case _ : raise Exception("This dataset configuration does not exist!")

def switch_equation(problem):
    match problem:
        case "Laplace1D": return eq.Laplace
        case "Laplace2D": return eq.Laplace
        case _ : raise Exception("This equation does not exist!")
