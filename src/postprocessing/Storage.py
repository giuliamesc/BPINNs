import numpy as np
import shutil
import json
import os

class Storage():
    """
    Class to manage import and saving of data
    Properties:
        - history
        - thetas
        - confidence
        - nn_samples
    Methods:
        - save_parameter
        - save_errors
    """
    def __init__(self, path_data, path_values, path_thetas, path_log):

        self.path_data   = path_data
        self.path_values = path_values
        self.path_thetas = path_thetas
        self.path_log    = path_log
        self.path_sample = os.path.join(self.path_values, "samples")
        self.idx_len = 3
        self.sg_flags = None

    @property
    def history(self):
        keys = ("Total", "res", "data", "prior")
        loss    = self.__load_dict(self.path_log, "loss.npy"   , keys)
        logloss = self.__load_dict(self.path_log, "logloss.npy", keys)
        return (loss, logloss)

    @history.setter
    def history(self, values):
        keys = ("Total", "res", "data", "prior")
        self.__save_dict(self.path_log, "loss.npy"   , keys, values[0])
        self.__save_dict(self.path_log, "logloss.npy", keys, values[1])

    @property
    def thetas(self):
        thetas = list()
        for folder in os.listdir(self.path_thetas):
            folder_path = os.path.join(self.path_thetas, folder)
            weights = self.__load_list(folder_path, "w", 2)
            biases  = self.__load_list(folder_path, "b", 2)
            theta = list()
            for w, b in zip(weights, biases):
                theta.append(w)
                theta.append(b)
            thetas.append(theta)
        return thetas

    @thetas.setter
    def thetas(self, values):
        self.__set_thetas_folder(len(values))
        for value, folder in zip(values, os.listdir(self.path_thetas)):
            folder_path = os.path.join(self.path_thetas, folder)
            self.__save_list(folder_path, "w", value[0::2], 2)
            self.__save_list(folder_path, "b", value[1::2], 2)

    @property 
    def sigmas(self):
        file_name_d = os.path.join(self.path_log, "sigma_d.npy")
        file_name_r = os.path.join(self.path_log, "sigma_r.npy")
        sigma_d = np.load(file_name_d) if os.path.isfile(file_name_d) else None
        sigma_r = np.load(file_name_r) if os.path.isfile(file_name_r) else None
        return sigma_d, sigma_r
    
    @sigmas.setter
    def sigmas(self, values):
        file_name_d = os.path.join(self.path_log, "sigma_d.npy")
        file_name_r = os.path.join(self.path_log, "sigma_r.npy")
        sigma_d = np.array([value[0][0].numpy() for value in values])
        sigma_r = np.array([value[0][1].numpy() for value in values])
        if self.sg_flags[0] : np.save(file_name_d, sigma_d)
        if self.sg_flags[1] : np.save(file_name_r, sigma_r)

    @property
    def data(self):
        dom_data = None
        fit_data = None
        return dom_data, fit_data

    @data.setter
    def data(self, dom_data, fit_data):
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

    def __write_txt_line(self, outfile, my_str, vec):
        comps = ["x", "y", "z"][:len(vec)]
        aux_str = my_str + " by component:" + "\n"
        aux_str = aux_str.ljust(20)
        for idx, comp in enumerate(comps):
            aux_str = aux_str + comp + ":"
            outfile.write(f"{aux_str} {vec[idx]:1.4f} \t")
        outfile.write("\n")

    def __write_par_line(self, outfile, field, value):
        outfile.write(field)
        max_len = max(len(key) for key in value.keys()) + 3
        for key, val in value.items():
            aux_str = "\t" + key + ":"
            aux_str = aux_str.ljust(max_len)
            outfile.write(f"{aux_str} {val} \n")
        outfile.write("\n")

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

    def __set_thetas_folder(self, num):

        shutil.rmtree(self.path_thetas)
        os.mkdir(self.path_thetas)
        file_path = os.path.join(self.path_thetas,"theta")+"_"
        
        for idx in range(num):
            folder_name = file_path + str(idx+1).zfill(self.idx_len)
            os.mkdir(folder_name)

    def __load_dict(self, path, name, keys):

        file_path = os.path.join(path, name)
        values_npy = np.load(file_path)

        value_dict = dict()
        for idx, key in enumerate(keys):
            value_dict[key] = list(values_npy[idx,:])

        return value_dict

    def __save_dict(self, path, name, keys, values):
        
        shape_np  = (len(keys),len(values[keys[0]]))
        values_np = np.zeros(shape_np, dtype=np.float32)

        for idx, key in enumerate(keys):
            values_np[idx,:] = values[key]

        file_path = os.path.join(path, name)
        np.save(file_path, values_np)

    def save_parameter(self, par):
        """Save parameters"""
        with open(os.path.join(self.path_log, "parameters.txt"), 'w') as outfile:
            general = dict(method = par.method, dataset = par.folder_name)
            self.__write_par_line(outfile, "GENERAL\n", general)
            self.__write_par_line(outfile, "EXPERIMENT\n", par.experiment)
            self.__write_par_line(outfile, "ARCHITECTURE\n", par.architecture)
            self.__write_par_line(outfile, "PARAMETERS\n", par.coeff)
            self.__write_par_line(outfile, "SIGMAS\n", par.sigmas)
            self.__write_par_line(outfile, "UTILS\n", par.utils)
            self.__write_par_line(outfile, str(par.method).upper() +"\n", par.param_method)

    def save_errors(self, errors):
        """Save errors"""
        with open(os.path.join(self.path_log, "errors.txt"), 'w') as outfile:
            outfile.write("RELATIVE ERRORS\n")
            self.__write_txt_line(outfile, "Solution", errors["error_sol"])
            self.__write_txt_line(outfile, "Parametetric field", errors["error_par"])
            outfile.write("\nUNCERTAINITY QUANTIFICATION (MEAN)\n")
            self.__write_txt_line(outfile, "Solution", errors["uq_sol_mean"])
            self.__write_txt_line(outfile, "Parametric field", errors["uq_par_mean"])
            outfile.write("\nUNCERTAINITY QUANTIFICATION (MAX)\n")
            self.__write_txt_line(outfile, "Solution", errors["uq_sol_max"])
            self.__write_txt_line(outfile, "Parametric field", errors["uq_par_max"])