from datetime import datetime
import tensorflow as tf

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
