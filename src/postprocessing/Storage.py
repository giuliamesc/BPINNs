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
        outfile.write(f" \"{msg}\": ")
        json.dump(value, outfile)
        outfile.write(", \n")

    def __write_txt_line(self, outfile, my_str, vec):
        comps = ["x", "y", "z"][:len(vec)]
        aux_str = my_str + " by component:" + "\n"
        aux_str = aux_str.ljust(20)
        for idx, comp in enumerate(comps):
            aux_str = aux_str + comp + ":"
            outfile.write(f"{aux_str} {vec[idx]:1.4f} \t")
        outfile.write("\n")

    def __write_par_line(self, outfile, field, value):
        outfile.write(field + "\n")
        max_len = max(len(key) for key in value.keys()) + 3
        for key, val in value.items():
            aux_str = key + ":"
            aux_str = aux_str.ljust(max_len)
            outfile.write(f"{aux_str} {val} \n")
        outfile.write("\n\n")

    def save_parameter(self, par):
        """Save parameters"""
        with open(os.path.join(self.path_log, "parameters.txt"), 'w') as outfile:
            self.__write_par_line(outfile, "ARCHITECTURE\n", par.architecture)
            self.__write_par_line(outfile, "EXPERIMENT\n", par.experiment)
            self.__write_par_line(outfile, "PARAMETERS\n", par.param)
            self.__write_par_line(outfile, "SIGMAS\n", par.sigmas)
            self.__write_par_line(outfile, str(par.method)+"\n", par.param_method)
            self.__write_par_line(outfile, "UTILS\n", par.utils)

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
        # print(file_path) Lo volevi stampare davvero?
        np.save(file_path, values_np)

    @property
    def losses(self):
        keys = ("Total", "res", "data", "prior")
        loss    = self.__load_dict(self.path_log, "loss.npy"   , keys)
        logloss = self.__load_dict(self.path_log, "logloss.npy", keys)
        return (loss, logloss)

    @losses.setter
    def losses(self, values):
        keys = ("Total", "res", "data", "prior")
        self.__save_dict(self.path_log, "loss.npy"   , keys, values[0])
        self.__save_dict(self.path_log, "logloss.npy", keys, values[1])


"""
values = (loss, logloss)
loss, logloss = {tot, res, data, prior}
tot, res, data, prior = [...]
"""