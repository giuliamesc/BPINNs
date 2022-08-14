import numpy as np
import shutil
import json
import os

class Storage():

    def __init__(self, path_values, path_thetas, path_log):

        self.path_values = path_values
        self.path_thetas = path_thetas
        self.path_log    = path_log
        self.path_sample = os.path.join(self.path_values, "samples")
        self.idx_len = 3

    def __write_line(self, outfile, msg, value):
        # USata solo per save params
        outfile.write(f" \"{msg}\": ")
        json.dump(value, outfile)
        outfile.write(", \n")

    def save_parameter(self, par):
        # CAMBIARE (GIULIA)
        """Save parameters"""
        with open(self.path_log, 'w') as outfile:
            outfile.write("{ \n")
            self.__write_line(outfile, "architecture", par.architecture)
            self.__write_line(outfile, "experiment", par.experiment)
            self.__write_line(outfile, "param", par.param)
            self.__write_line(outfile, "sigmas", par.sigmas)
            self.__write_line(outfile, str(par.method), par.param_method)
            self.__write_line(outfile, "utils", par.utils)
            outfile.write("}")

    def save_errors(self, errors):
        # CAMBIARE (GIULIA), NELLA CARTELLA PATH_LOG
        pass

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

    def __load_list(self, path, name, num_len):
        outputs = list()
        for file_value in os.listdir(path):
            if not file_value[:-(1+num_len+4)] == name: continue
            outputs.append(np.load(os.path.join(self.path_sample, file_value))) 
        return outputs

    def __save_list(self, path, name, values, num_len):
        file_path = os.path.join(path,name)+"_"
        for idx, value in enumerate(values):
            file_name = file_path + str(idx+1).zfill(num_len) + ".npy"
            np.save(file_name, value)

    @property
    def nn_samples(self):
        functions_nn_samples = {
            "sol_samples": self.__load_list(self.path_sample, "sol", self.idx_len), 
            "par_samples": self.__load_list(self.path_sample, "par", self.idx_len)}
        return functions_nn_samples

    @nn_samples.setter
    def nn_samples(self, values):
        self.__save_list(self.path_sample, "sol", values["sol_samples"], self.idx_len)
        self.__save_list(self.path_sample, "par", values["par_samples"], self.idx_len)

    def __set_thetas_folder(self, num):

        shutil.rmtree(self.path_thetas)
        os.mkdir(self.path_thetas)
        file_path = os.path.join(self.path_thetas,"theta")+"_"
        
        for idx in range(num):
            folder_name = file_path + str(idx+1).zfill(self.idx_len)
            os.mkdir(folder_name)

    @property
    def thetas(self):
        thetas = list()
        for folder in os.listdir(self.path_thetas):
            folder_path = os.path.join(self.path_thetas, folder)
            weights = self.__load_list(folder_path, "w", 2)
            biases  = self.__load_list(folder_path, "b", 2)
            thetas.append((weights,biases))
        return thetas

    @thetas.setter
    def thetas(self, values):
        self.__set_thetas_folder(len(values))
        for value, folder in zip(values, os.listdir(self.path_thetas)):
            folder_path = os.path.join(self.path_thetas, folder)
            self.__save_list(folder_path, "w", value[0], 2)
            self.__save_list(folder_path, "b", value[1], 2)

    @property
    def losses(self):
        pass
    @losses.setter
    def losses(self, values):
        pass
