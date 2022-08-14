import algorithms

def switch_algorithm(method, test = False):
    if   test : return algorithms.Test_Alg
    if   method == "VI"   : return algorithms.VI
    elif method == "HMC"  : return algorithms.HMC
    elif method == "SVGD" : return algorithms.SVGD
    else : raise Exception("This algorithm does not exist!")