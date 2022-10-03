from genericpath import isdir
import json
import os
import logging

def set_config(default, hard_coded):
    return default if hard_coded is None else hard_coded

def set_directory():
    """ Sets the working directory """
    if os.getcwd()[-3:] != "src":
        new_dir = os.path.join(os.getcwd(),"src")
        os.chdir(new_dir)
        print(f"Working Directory moved to: {new_dir}")

def set_warning():
    """ Sets the level of warning to print """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    logging.basicConfig(level=logging.ERROR)

def compute_gui_len():
    return max(80,int(os.get_terminal_size().columns/2))

def starred_print(message):
    gui_len = compute_gui_len()
    print(f" {message} ".center(gui_len,'*'))

def load_json(path):
    """ Load the json file with all the parameters """
    with open(os.path.join("../config", path +".json")) as hpFile:
        hp = json.load(hpFile)
    return hp

def check_dataset(data_config):
    path = os.path.join("../data", data_config.problem)
    path = os.path.join(path, data_config.name)
    if os.path.isdir(path): return
    raise Exception(f"Dataset {data_config.name} doesn't exists! (Maybe need to generate it)")
