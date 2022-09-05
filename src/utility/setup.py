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

def set_gui_len():
    """ Sets terminal width """
    return max(50,int(os.get_terminal_size().columns/3))

def load_json(path):
    """ Load the json file with all the parameters """
    with open(os.path.join("../config", path +".json")) as hpFile:
        hp = json.load(hpFile)
    return hp