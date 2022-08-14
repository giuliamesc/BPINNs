import numpy as np
import json
import os

class Storage():

    def __init__(self, path_values, path_weights):

        self.path_values = path_values
        self.path_weights = path_weights
        self.path_sample = os.path.join(self.path_values, "samples")
        self.idx_len = 3

    def __write_line(self, outfile, msg, value):

        outfile.write(f" \"{msg}\": ")
        json.dump(value, outfile)
        outfile.write(", \n")

    def save_parameter(self, par):
        """Save parameters"""
        with open(self.path_weights, 'w') as outfile:
            outfile.write("{ \n")
            self.__write_line(outfile, "architecture", par.architecture)
            self.__write_line(outfile, "experiment", par.experiment)
            self.__write_line(outfile, "param", par.param)
            self.__write_line(outfile, "sigmas", par.sigmas)
            self.__write_line(outfile, str(par.method), par.param_method)
            self.__write_line(outfile, "utils", par.utils)
            outfile.write("}")

    @property
    def confidence(self):
        functions_confidence = {
            "sol_NN" : np.load(os.path.join(self.path_values, "sol_NN.npy" )), 
            "sol_std": np.load(os.path.join(self.path_values, "sol_std.npy")), 
            "par_NN" : np.load(os.path.join(self.path_values, "par_NN.npy" )), 
            "par_std": np.load(os.path.join(self.path_values, "par_std.npy"))}
        return functions_confidence

    @confidence.setter
    def confidence(self, values):
        np.save(os.path.join(self.path_values, "sol_NN.npy" ), values["sol_NN" ])
        np.save(os.path.join(self.path_values, "sol_std.npy"), values["sol_std"])
        np.save(os.path.join(self.path_values, "par_NN.npy" ), values["par_NN" ])
        np.save(os.path.join(self.path_values, "par_std.npy"), values["par_std"])

    def __load_list(self, path, name):
        outputs = list()
        for file_value in os.listdir(path):
            if not file_value[:-(1+self.idx_len+4)] == name: continue
            outputs.append(np.load(os.path.join(self.path_sample, file_value))) 
        return outputs

    def __save_list(self, path, name, values):
        file_path = os.path.join(path,name)+"_"
        for idx, value in enumerate(values):
            file_name = file_path + str(idx).zfill(self.idx_len) + ".npy"
            np.save(file_name, value)

    @property
    def nn_samples(self):
        functions_nn_samples = {
            "sol_samples": self.__load_list(self.path_sample, "sol"), 
            "par_samples": self.__load_list(self.path_sample, "par")}
        return functions_nn_samples

    @nn_samples.setter
    def nn_samples(self, values):
        self.__save_list(self.path_sample, "sol", values["sol_samples"])
        self.__save_list(self.path_sample, "par", values["par_samples"])

    def save_training(self, theta, loss):
        pass

    def save_errors(self, errors):
        pass

    def load_losses(self):
        pass

    def load_errors(self):
        pass