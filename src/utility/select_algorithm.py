import algorithms

def switch_algorithm(method):
    """ Returns an instance of the class corresponding to the selected method """
    match method:
        case "TEST": return algorithms.TEST
        case "HMC" : return algorithms.HMC
        case "SVGD": return algorithms.SVGD
        case "VI"  : return algorithms.VI
        case _ : raise Exception("This algorithm does not exist!")