import os
import shutil
from datetime import datetime

def __create_keeper(path):
    """ Creation of the .gitkeep file """
    file_name = os.path.join(path,".gitkeep")
    open(file_name, 'w')

def __create_single_dir(base_path, last_path, keep = False, over = False):
    """ Creation of a single directory """
    folder_path = os.path.join(base_path, last_path)
    try: os.makedirs(folder_path)
    except: 
        if over:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
    if keep: __create_keeper(folder_path) 
    return folder_path

def create_directories(par):
    """ Create all the directories we need to store the results """
    
    problem_folder   = __create_single_dir("../outs", par.problem, keep=True)
    case_folder = __create_single_dir(problem_folder, par.folder_name, keep=True)
    case_time = f"{datetime.now().strftime('%Y.%m.%d-%H.%M.%S')}"
    file_name = par.method + "_" + case_time if par.utils["save_flag"] else "trash"
    case_folder = __create_single_dir(case_folder, file_name, over=True)
    
    path_plot   = __create_single_dir(case_folder, "plot")
    path_data   = __create_single_dir(case_folder, "data")
    path_values = __create_single_dir(case_folder, "values")
    path_thetas = __create_single_dir(case_folder, "thetas")
    path_log    = __create_single_dir(case_folder, "log")
    __create_single_dir(path_values, "samples")

    return path_plot, path_data, path_values, path_thetas, path_log

def create_data_folders(problem, name, save):
    problem_folder  = __create_single_dir("../data", problem, keep=True)
    if not save: name = "trash"
    case_folder     = __create_single_dir(problem_folder, name, over=True)
    return case_folder
