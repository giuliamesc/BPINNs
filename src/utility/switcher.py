import algorithms as alg
import datasets   as data
import equations  as eq

def switch_algorithm(method):
    """ Returns an instance of the class corresponding to the selected method """
    match method:
        case "ADAM": return alg.ADAM
        case "HMC" : return alg.HMC
        case "SVGD": return alg.SVGD
        case "VI"  : return alg.VI
        case _ : raise Exception("This algorithm does not exist!")

def switch_dataset(problem, case_name):
    match problem:
        case "Regression":
            match case_name:
                case "cos": return data.Reg1D_cos()
                case "sin": return data.Reg1D_sin()
                case _ :  raise Exception("This case test does not exist!")
        case "Laplace1D": 
            match case_name:
                case "cos": return data.Laplace1D_cos()
                case "sin": return data.Laplace1D_sin()
                case _ :  raise Exception("This case test does not exist!")
        case "Oscillator":
            match case_name:
                case "sin": return data.Oscillator1D()
                case _ :  raise Exception("This case test does not exist!")
        case "Laplace2D": raise Exception("Not implemeted yet")
        case _ : raise Exception(f"This dataset configuration does not exist: {problem}")

def switch_equation(equation):
    match equation:
        case "Regression": return eq.Regression
        case "Laplace1D":  return eq.Laplace
        case "Laplace2D":  return eq.Laplace
        case "Oscillator": return eq.Oscillator
        case _ : raise Exception(f"This equation does not exist: {equation}")
