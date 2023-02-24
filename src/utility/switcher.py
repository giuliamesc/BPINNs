import datasets   as data
import equations  as eq

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
        case "Laplace2D": 
            match case_name:
                case "cos": return data.Laplace2D_cos()
                case _ :  raise Exception("This case test does not exist!")
        case _ : raise Exception(f"This dataset configuration does not exist: {problem}")

def switch_equation(equation):
    match equation:
        case "Regression": return eq.Regression
        case "Laplace1D":  return eq.Laplace
        case "Laplace2D":  return eq.Laplace
        case "Oscillator": return eq.Oscillator
        case _ : raise Exception(f"This equation does not exist: {equation}")

def switch_configuration(name, test_mode=False):
    test_cases = [None, "ADAM_oscillator", "ADAM_regression", "ADAM_laplace", "HMC_regression", "HMC_laplace", 
                    "SVGD_oscillator", "VI_regression"]
    best_cases = [None, "ADAM_lap_cos", "HMC_lap_cos", "HMC_reg_cos", "HMC_reg_sin", "ADAM_oscillator"]

    config_folder = "test_models/" if test_mode else "best_models/"
    config_list   =  test_cases    if test_mode else  best_cases
    config_file   = config_list[name] if type(name) == int else name

    return config_folder + config_file