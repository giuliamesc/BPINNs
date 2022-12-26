from utility import create_paths
import numpy as np
import shutil
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
    def __init__(self, path_folder):

        paths = create_paths(path_folder)
        self.path_data   = paths[1]
        self.path_values = paths[2]
        self.path_thetas = paths[3]
        self.path_log    = paths[4]
        self.path_sample = os.path.join(self.path_values, "samples")
        self.idx_len = 3

    @property
    def history(self):
        keys = list()
        with open(os.path.join(self.path_log,"keys.txt"),'r') as fk: 
            for line in fk: keys.append(line[:-1])
        posterior     = self.__load_dict(self.path_log, "posterior.npy"    , keys)
        loglikelihood = self.__load_dict(self.path_log, "loglikelihood.npy", keys)
        return (posterior, loglikelihood)

    @history.setter
    def history(self, values):
        keys = sorted(list(values[0].keys()))
        with open(os.path.join(self.path_log,"keys.txt"),'w') as fk: fk.write('\n'.join(keys)+"\n")
        self.__save_dict(self.path_log, "posterior.npy"    , keys, values[0])
        self.__save_dict(self.path_log, "loglikelihood.npy", keys, values[1])

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
            self.__save_list(folder_path, "w", value.weights, 2)
            self.__save_list(folder_path, "b", value.biases,  2)

    @property
    def data(self):
        load_dom = lambda x: np.load(os.path.join(self.path_data, f"{x}_dom.npy"))
        load_val = lambda x: np.load(os.path.join(self.path_data, f"{x}_val.npy"))
        names = ["sol_ex", "par_ex", "sol_ns", "par_ns", "bnd_ns"]    
        plots = {k: (load_dom(k),load_val(k)) for k in names}
        return plots

    @data.setter
    def data(self, data_plot):
        save_dom = lambda x: np.save(os.path.join(self.path_data, f"{x}_dom.npy"), data_plot[x][0])
        save_val = lambda x: np.save(os.path.join(self.path_data, f"{x}_val.npy"), data_plot[x][1])
        names = ["sol_ex", "par_ex", "sol_ns", "par_ns", "bnd_ns"] 
        for k in names: 
            save_dom(k) 
            save_val(k)

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
        outfile.write(aux_str)
        for idx, comp in enumerate(comps):
            outfile.write(f"\t{comp}: {100*vec[idx]:1.2f}%")
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

        for idx, key, in enumerate(keys):
            values_np[idx,:] = values[key]

        file_path = os.path.join(path, name)
        np.save(file_path, values_np)

    def save_parameter(self, par):
        """Save parameters"""
        with open(os.path.join(self.path_log, "parameters.txt"), 'w') as outfile:
            general = dict(method = par.method, problem = par.problem, case_name = par.case_name)
            self.__write_par_line(outfile, "GENERAL\n", general)
            self.__write_par_line(outfile, "ARCHITECTURE\n", par.architecture)
            self.__write_par_line(outfile, "NUM_POINTS\n", par.num_points)
            self.__write_par_line(outfile, "UNCERTAINTY\n", par.uncertainty)
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