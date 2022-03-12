#
import json
import os

# local import
from helpers import string_to_bool

"""
Class to handle all the parameters
"""
class param:
    """Initializer"""
    def __init__(self, hp, args):
        ## method used: SVGD or HMC
        self.method = args.method
        ## NN architecture param
        self.architecture = hp["architecture"]
        ## experiment param
        self.experiment = hp["experiment"]
        ## general param
        self.param = hp["param"]
        ## sigmas param
        self.sigmas = hp["sigmas"]
        ## utils param
        self.utils = hp["utils"]

        ## specific param for the selected method
        self.param_method = hp[args.method]

        ## convert all the string "True" or "False" to boolean
        self._change_string_to_bool()

        # if we have some additional parameters from the command-line
        if(self._length_additional(vars(args)) > 0):
            self._update(vars(args))    # update param overspecified by command-line

        # for these parameters we use the dictionaries specified at the bottom (n_input, pde and dataset_type)
        ## store the dimension input (1D, 2D or 3D)
        self.n_input = n_input[self.experiment["dataset"]]
        ## isotropic or anisotropic
        self.pde = pde[self.experiment["dataset"]]
        ## analytical or dataset
        self.dataset_type = dataset_type[self.experiment["dataset"]]

        ## dimension of velocity: 1 if isotropic, >1 if anisotropic
        self.n_output_vel = 1
        if(self.experiment["dataset"]=="anisotropic1"):
            self.n_output_vel = 3
        if(self.experiment["dataset"]=="anisotropic2"):
            self.n_output_vel = 2

        # if experiment is analytical, specify the n of domain points
        if(self.dataset_type == "analytical"):
            if self.n_input == 1:
                self.experiment["n_domain"] = 1000
            elif self.n_input == 2:
                self.experiment["n_domain"] = 10000
            else:
                self.experiment["n_domain"] = 1000000

        # check possible errors in parameters
        self._check_parameter()

################################################################################

    def _length_additional(self, args_dict):
        """Compute the lenght of additional parameters specified by command-line"""
        i = 0
        for key in args_dict:
            if(args_dict[key] != None):
                i+=1
        return (i-2)    # the first 2 are mandatory (method and config) and dont count as additional param


    def _change_string_to_bool(self):
        """Change "True" and "False" string to boolean for each bool parameter """
        self.experiment["is_uniform_exact"] = string_to_bool(self.experiment["is_uniform_exact"])
        self.sigmas["data_prior_noise_trainable"] = string_to_bool(self.sigmas["data_prior_noise_trainable"])
        self.sigmas["pde_prior_noise_trainable"] = string_to_bool(self.sigmas["pde_prior_noise_trainable"])
        self.utils["verbose"] = string_to_bool(self.utils["verbose"])
        self.utils["save_flag"] = string_to_bool(self.utils["save_flag"])


    def _update(self, args_dict):
        """Update the parameter given by json file using args (overspecification by command-line)"""
        i = 0
        for key in args_dict:
            if(i > 2):
                if args_dict[key] != None:
                    if key in self.architecture:
                        self.architecture[key] = args_dict[key]
                    elif key in self.experiment:
                        self.experiment[key] = args_dict[key]
                    elif key in self.param:
                        self.param[key] = args_dict[key]
                    elif key in self.sigmas:
                        self.sigmas[key] = args_dict[key]
                    elif key in self.utils:
                        self.utils[key] = args_dict[key]
                    else: #param_method...
                        if key in self.param_method:
                            self.param_method[key] = args_dict[key]
                        else:
                            print("Wrong parameter ", key," for the selected method: ", self.method)
            i+=1

    def _check_parameter(self):
        """Check the parameters"""
        if(self.method != "HMC" and self.method != "SVGD"):
            raise Exception("method not supported")

        if (not isinstance(self.architecture["n_layers"], int) or not isinstance(self.architecture["n_neurons"],int)):
            raise TypeError("n_layers and n_neurons must be integer")

        if (not isinstance(self.experiment["prop_exact"], float) or not isinstance(self.experiment["prop_collocation"],float)):
            raise TypeError("prop_coll and prop_exact must be float")
        if self.experiment["prop_exact"]<0 or self.experiment["prop_exact"]>1:
            raise Exception("Prop exact must be between 0 and 1")
        if self.experiment["prop_collocation"]<0 or self.experiment["prop_collocation"]>1:
            raise Exception("Prop coll must be between 0 and 1")

        if (not isinstance(self.experiment["noise_lv"],float)):
            raise TypeError("noise level must be float")
        if self.experiment["noise_lv"]<0:
            raise Exception("noise level must be >= 0")
        if self.experiment["batch_size"]<0:
            raise Exception("Batch size must be >=0")

        if self.sigmas["data_prior_noise"]<0:
            raise Exception("data_prior_noise must be >= 0")
        if self.sigmas["pde_prior_noise"]<0:
            raise Exception("pde_prior_noise must be >= 0")

        if self.method == "SVGD":
            if self.param_method["lr"]<=0:
                raise Exception("learning rate must be > 0")
            if self.param_method["lr_noise"]<=0:
                raise Exception("learning rate for noise must be > 0")
            if self.param_method["param_repulsivity"]<=0:
                raise Exception("param_repulsivity must be > 0")
            if (not isinstance(self.param_method["n_samples"], int)):
                raise TypeError("N samples must be integer")
            if self.param_method["n_samples"]<=0:
                raise Exception("N samples must be positive integer")
            if (not isinstance(self.param_method["epochs"], int)):
                raise TypeError("N samples must be integer")
            if self.param_method["epochs"]<=0:
                raise Exception("epochs must be a positive integer")

        if self.method == "HMC":
            if self.param_method["dt_HMC"]<=0:
                raise Exception("dt must be > 0")
            if self.param_method["dt_noise_HMC"]<=0:
                raise Exception("dt for noise must be > 0")
            if not( isinstance(self.param_method["N_HMC"],int) and isinstance(self.param_method["M_HMC"],int)
                    and isinstance(self.param_method["L_HMC"],int) ):
                raise TypeError("N_HMC, M_HMC and L_HMC must be integers")
            if self.param_method["N_HMC"]<0 or self.param_method["M_HMC"]<0 or self.param_method["N_HMC"]<self.param_method["M_HMC"]:
                raise Exception(" problem in definition of N and M ")



    def print_parameter(self):
        """Print all the parameters"""
        print("Method: ", self.method, " \n ")
        print("architecture: ", self.architecture, " \n ")
        print("experiment: ", self.experiment, " \n ")
        print("param: ", self.param, " \n ")
        print("sigmas: ", self.sigmas, " \n ")
        print("utils: ", self.utils, " \n ")
        print("param_method: ", self.param_method, " \n ")
        print("n_input: ", self.n_input, " \n ")
        print("n_output_vel: ", self.n_output_vel, " \n ")
        print("pde_type: ", self.pde, " \n ")
        print("dataset_type: ", self.dataset_type)


    def save_parameter(self, path=""):
        """Save parameters"""
        with open(os.path.join(path,'param.json'), 'w') as outfile:
            outfile.write("{ \n")

            outfile.write(" \"architecture\": ")
            json.dump(self.architecture, outfile)
            outfile.write(", \n")

            outfile.write(" \"experiment\": ")
            json.dump(self.experiment, outfile)
            outfile.write(", \n")

            outfile.write(" \"param\": ")
            json.dump(self.param, outfile)
            outfile.write(", \n")

            outfile.write(" \"sigmas\": ")
            json.dump(self.sigmas, outfile)
            outfile.write(", \n")

            s = " \""+self.method+"\": "
            outfile.write(s)
            json.dump(self.param_method, outfile)
            outfile.write(", \n")

            outfile.write(" \"utils\": ")
            json.dump(self.utils, outfile)
            outfile.write("\n")

            outfile.write("}")

"""dictionary for input dimension given the dataset used"""
n_input = {
"elliptic_cos": 1
}


"""dictionary for isotropic or anisotropic pde given the dataset used"""
pde = {
"elliptic_cos": "laplace"
}


"""dictionary for dataset_type (analytical functions or real dataset from Pykonal) given the dataset used"""
dataset_type = {
"elliptic_cos": "dataset"
}
