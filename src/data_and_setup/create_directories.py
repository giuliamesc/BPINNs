import os
from datetime import datetime

def create_directories(par):
    """!
    Create all the directories we need to store the results

    @param par an object of param class (parameters)
    """
    save_flag = par.utils["save_flag"]
    n_input = par.n_input
    dataset_name = par.experiment["dataset"]
    pde_type = par.pde
    method_name = par.method

    case_name = "../" + str(n_input)+"D-"+pde_type

    if(save_flag):
        ## if save_flag = True create new directories using datetime.now()
        now = datetime.now()
        path = method_name + "_" + f"{now.strftime('%Y.%m.%d-%H.%M.%S')}"
        ## path result
        path_result = os.path.join(case_name, path)
        os.makedirs(path_result)
        ## path_plot
        path_plot = os.path.join(path_result, "plot")
        os.makedirs(path_plot)
        ## path_weights
        path_weights = os.path.join(path_result, "weights")
        os.makedirs(path_weights)
    else:
        ## if save_flag = False store everything in a directories named "trash" that will be overwritten everytime
        path_result = os.path.join(case_name, "trash")
        try: os.makedirs(path_result)
        except: pass
        path_plot = os.path.join(path_result, "plot")
        try: os.makedirs(path_plot)
        except: pass
        path_weights = os.path.join(path_result, "weights")
        try: os.makedirs(path_weights)
        except: pass

    return path_result, path_plot, path_weights
