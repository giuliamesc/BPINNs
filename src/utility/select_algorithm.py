import algorithms

def switch_algorithm(par, test = False):
    if   test : return algorithms.Test_Alg
    if   par.method == "VI"   : return algorithms.VI
    elif par.method == "HMC"  : return algorithms.HMC
    elif par.method == "SVGD" : return algorithms.SVGD
    else : raise Exception("This algorithm does not exist!")