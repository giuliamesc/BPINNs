class Param:
    """Initializer"""
    def __init__(self, hp, args):

        self.architecture = hp["architecture"] # NN architecture param
        self.experiment   = hp["experiment"]   # experiment param
        
        self.param  = hp["param"]  # general param
        self.sigmas = hp["sigmas"] # sigmas param
        self.utils  = hp["utils"]  # utils param

        self.method = args.method ## method used: SVGD, HMC, VI, ...
        self.param_method = hp[args.method] # specific param for the selected method

        # if we have some additional parameters from the command-line
        self.__command_line_update(vars(args))
        # set all useful parameters from physical domain and dimension
        self.phys_dim = Dimension(self.experiment["dataset"], True)
        self.comp_dim = Dimension(self.experiment["dataset"], False)
        self.pde = self.phys_dim.pde[self.experiment["dataset"]]

    def __string_to_bool(self, s):
        """
        Convert string "True","False" to boolean True and False
        """
        if s=="False" or s=="false": return False
        elif s=="True" or s=="true": return True
        else: print("no boolean string")

    def __change_string_to_bool(self):
        """Change "True" and "False" string to boolean for each bool parameter """
        self.sigmas["data_prior_noise_trainable"] = self.__string_to_bool(self.sigmas["data_prior_noise_trainable"])
        self.sigmas["pde_prior_noise_trainable"]  = self.__string_to_bool(self.sigmas["pde_prior_noise_trainable"])
        self.utils["save_flag"]  = self.__string_to_bool(self.utils["save_flag"])
        self.utils["debug_flag"] = self.__string_to_bool(self.utils["debug_flag"])

    def __command_line_update(self, args_dict):
        """Update the parameter given by json file using args (overspecification by command-line)"""
        self.__change_string_to_bool() # convert all the string "True" or "False" to boolean
        self.json_dict = [self.architecture, self.experiment, self.param, self.sigmas, self.utils]
        for key in args_dict:
            if args_dict[key] is None: continue
            for jdict in self.json_dict:
                if key in jdict: jdict[key] = args_dict[key]
            if key in self.param_method:
                self.param_method[key] = args_dict[key]
        

class Dimension():
    
    def __init__(self, problem, physical):

        self.__def_params()
        self.dimensions = self.phys_dom if physical else self.comp_dom

        ## store the dimension input (1D, 2D or 3D)
        self.n_input   = self.dimensions[problem][0]
        ## dimension of solution
        self.n_out_sol = self.dimensions[problem][1]
        ## dimension of parametric field
        self.n_out_par = self.dimensions[problem][2]

    def __def_params(self):

        """dictionary for pde given the dataset used"""
        self.pde = {
        "laplace1D_cos": "laplace",
        "laplace2D_cos": "laplace"
        }

        """dictionary for I/O dimensions of the dataset used"""
        self.phys_dom = {
        "laplace1D_cos": (1,1,1),
        "laplace2D_cos": (2,1,1)
        }

        """dictionary for I/O dimensions of the network used"""
        self.comp_dom = {
        "laplace1D_cos": (1,1,1),
        "laplace2D_cos": (2,1,1)
        }
