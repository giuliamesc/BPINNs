import os
from datetime import datetime

def create_keeper(path):
    file_name = os.path.join(path,".gitkeep")
    open(file_name, 'w')

def create_single_dir(base_path, last_path, keep = False):
    
    folder_path = base_path if last_path is None else os.path.join(base_path, last_path)
    
    try: os.makedirs(folder_path)
    except: pass

    if keep: create_keeper(folder_path) 
    
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
    path_case = os.path.join("../outs",case_name)
    create_single_dir(path_case, None, keep = True)

    if save_flag:
        ## if save_flag = True create new directories using datetime.now()
        now = datetime.now()
        path_test = method_name + "_" + f"{now.strftime('%Y.%m.%d-%H.%M.%S')}"
        path_result = create_single_dir(path_case, path_test)
    else:
        ## if save_flag = False store everything in a directories named "trash" that will be overwritten everytime
        path_result = create_single_dir(path_case,"trash")

    path_plot   = create_single_dir(path_result, "plot")
    path_values = create_single_dir(path_result, "values")
    path_thetas = create_single_dir(path_result, "thetas")
    path_log    = create_single_dir(path_result, "log")
    create_single_dir(path_values, "samples")

    return path_plot, path_values, path_thetas, path_log
