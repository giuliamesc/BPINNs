import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

import copy
import math
from datetime import datetime

import os
import psutil

"""
Define some useful functions that we use in the code
"""

def memory():
    """
    Compute the memory used
    """
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB
    return memoryUse

def string_to_bool(s):
    """
    Convert string "True","False" to boolean True and False
    """
    if s=="False" or s=="false":
        return False
    elif s=="True" or s=="true":
        return True
    else:
        print("no boolean string")

def get_trainable_weights_flatten(grad_parameters):
    """!
    For SVGD alg.
    flatten the list of list of parameters in input into a tensor of shape (num_neural_networks, lenght_of_theta)
    """
    w = []
    ## for loop over num_neural_networks
    for i in range( len(grad_parameters) ):
        w_i = []
        ## for loop on list of parameter
        for param in grad_parameters[i] :
            ## reshape
            w_i.append(tf.reshape(param,[-1]))
        w.append(tf.concat(w_i, axis=0))

    ## return a tensor of shape=(num_neural_networks, num_total_parameters_theta)
    return tf.convert_to_tensor(w)


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

    case_name = str(n_input)+"D-"+pde_type+"-eikonal"

    if(save_flag):
        ## if save_flag = True create new directories using datetime.now()
        now = datetime.now()
        path = dataset_name + "_" + method_name + "_" + f"{now.strftime('%Y%m%d-%H%M%S')}"
        ## path result
        path_result = os.path.join(case_name,"results", path)
        os.makedirs(path_result)
        ## path_plot
        path_plot = os.path.join(path_result, "plot")
        os.makedirs(path_plot)
        ## path_weights
        path_weights = os.path.join(path_result, "weights")
        os.makedirs(path_weights)
    else:
        ## if save_flag = False store everything in a directories named "trash" that will be overwritten everytime
        path_result = os.path.join(case_name,"results", "trash")
        try:
            os.makedirs(path_result)
        except:
            pass
        path_plot = os.path.join(path_result, "plot")
        try:
            os.makedirs(path_plot)
        except:
            pass
        path_weights = os.path.join(path_result, "weights")
        try:
            os.makedirs(path_weights)
        except:
            pass

    return path_result, path_plot, path_weights
