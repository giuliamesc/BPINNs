import algorithms

def switch_algorithm(method):
    """ Returns an instance of the class corresponding to the selected method """
    if   method == "TEST" : return algorithms.TEST
    elif method == "HMC"  : return algorithms.HMC
    elif method == "SVGD" : return algorithms.SVGD
    elif method == "VI"   : return algorithms.VI
    else : raise Exception("This algorithm does not exist!")