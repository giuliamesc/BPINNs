import os
from datetime import datetime

def create_keeper(path):
    file_name = os.path.join(path,".gitkeep")
    open(file_name, 'x')

def create_single_dir(base_path, last_path, keep = False):
    
    try: os.makedirs(os.path.join(base_path, last_path))
    except: pass

    folder_path = os.path.join(base_path, last_path)
    if keep: os.path.join(base_path, last_path)
    
    return folder_path

def create_directories(par):
    """!
    Create all the directories we need to store the results

    @param par an object of param class (parameters)
    """
    save_flag = par.utils["save_flag"]
    n_input  = par.phys_dim.n_input
    pde_type = par.pde
    method_name = par.method

    case_name = str(n_input)+"D-"+pde_type
    path_case = os.path.join("../results",case_name)
    os.makedirs(path_case)
    create_keeper(path_case)

    if save_flag:
        ## if save_flag = True create new directories using datetime.now()
        now = datetime.now()
        path_test = method_name + "_" + f"{now.strftime('%Y.%m.%d-%H.%M.%S')}"
        path_result = create_single_dir(path_case, path_test)
    else:
        ## if save_flag = False store everything in a directories named "trash" that will be overwritten everytime
        path_result = create_single_dir(path_case,"trash")

    path_plot    = create_single_dir(path_result, "plot")
    path_values  = create_single_dir(path_result, "values")
    path_weights = create_single_dir(path_result, "weights")
    create_single_dir(path_values, "samples")

    return path_plot, path_values, path_weights
