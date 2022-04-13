import os
from datetime import datetime

def create_single_dir(base_path, last_path):
    try: os.makedirs(base_path)
    except: pass
    return os.path.join(base_path, last_path)

def create_directories(par):
    """!
    Create all the directories we need to store the results

    @param par an object of param class (parameters)
    """
    save_flag = par.utils["save_flag"]
    n_input = par.n_input
    pde_type = par.pde
    method_name = par.method

    case_name = str(n_input)+"D-"+pde_type
    path_case = os.path.join("../results",case_name)

    if(save_flag):
        ## if save_flag = True create new directories using datetime.now()
        now = datetime.now()
        path_test = method_name + "_" + f"{now.strftime('%Y.%m.%d-%H.%M.%S')}"
        path_result = create_single_dir(path_case, path_test)
    else:
        ## if save_flag = False store everything in a directories named "trash" that will be overwritten everytime
        path_result = create_single_dir(path_case,"trash")
    path_plot    = create_single_dir(path_result, "plot")
    path_weights = create_single_dir(path_result, "weights")

    return path_result, path_plot, path_weights
