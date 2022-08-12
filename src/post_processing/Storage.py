import json
import os

class Storage():
    def __init__(self, path_result, path_plot, path_weights):
        self.path_result = path_result
        self.path_plot = path_plot
        self.path_weights = path_weights
        pass

    def save_parameter(self, par):
        """Save parameters"""
        with open(os.path.join(self.path_result,'param.json'), 'w') as outfile:
            outfile.write("{ \n")

            outfile.write(" \"architecture\": ")
            json.dump(par.architecture, outfile)
            outfile.write(", \n")

            outfile.write(" \"experiment\": ")
            json.dump(par.experiment, outfile)
            outfile.write(", \n")

            outfile.write(" \"param\": ")
            json.dump(par.param, outfile)
            outfile.write(", \n")

            outfile.write(" \"sigmas\": ")
            json.dump(par.sigmas, outfile)
            outfile.write(", \n")

            s = " \""+par.method+"\": "
            outfile.write(s)
            json.dump(par.param_method, outfile)
            outfile.write(", \n")

            outfile.write(" \"utils\": ")
            json.dump(par.utils, outfile)
            outfile.write("\n")

            outfile.write("}")

    def save_training(self, theta, loss):
        pass

    def save_results(self, functions_confidence, functions_nn_samples):
        pass

    def save_errors(self, errors):
        pass

    def load_losses(self):
        pass
    
    def load_confidence(self):
        pass

    def load_nn_samples(self):
        pass

    def load_errors(self):
        pass