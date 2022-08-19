class Param:
    """Initializer"""
    def __init__(self, hp, args):

        self.dataset = hp["general"]["dataset"] # dataset used
        self.method  = hp["general"]["method"]  # method used: SVGD, HMC, VI, ...

        self.experiment   = hp["experiment"]   # experiment param
        self.architecture = hp["architecture"] # NN architecture param
        
        self.coeff  = hp["coeff"]  # coefficient param
        self.sigmas = hp["sigmas"] # sigmas param
        self.utils  = hp["utils"]  # utilities param

        # if we have some additional parameters from the command-line
        self.__command_line_update_params(vars(args))

        # specific param for the selected method
        self.param_method = hp[self.method]
        self.__command_line_update_method(vars(args))

        # set all useful parameters from physical domain and dimension
        self.phys_dim = Dimension(self.dataset, True)
        self.comp_dim = Dimension(self.dataset, False)
        self.pde = self.phys_dim.pde[self.dataset]

        # Convert string of boolean in boolean
        self.__change_string_to_bool()

    def __string_to_bool(self, s):
        """ Convert string "True","False" to boolean True and False """
        if   s=="False" or s=="false": return False
        elif s=="True"  or s=="true" : return True
        else: raise Exception("No boolean string!")

    def __change_string_to_bool(self):
        """ Change "True" and "False" string to boolean for each bool parameter """
        self.sigmas["data_pn_flag"] = self.__string_to_bool(self.sigmas["data_pn_flag"])
        self.sigmas["pde_pn_flag"]  = self.__string_to_bool(self.sigmas["pde_pn_flag"])
        self.utils["save_flag"]  = self.__string_to_bool(self.utils["save_flag"])
        self.utils["debug_flag"] = self.__string_to_bool(self.utils["debug_flag"])

    def __command_line_update_params(self, args_dict):
        """ Update the parameter given by json file using args (overspecification by command-line) """
        for key in args_dict:
            if args_dict[key] is None: continue
            if key == "dataset" : self.dataset = args_dict[key]
            if key == "method"  : self.method  = args_dict[key]
            for jdict in [self.experiment, self.architecture, self.coeff, self.sigmas, self.utils]:
                if key in jdict: jdict[key] = args_dict[key]

    def __command_line_update_method(self, args_dict):
        """ Update the parameter given by json file using args (overspecification by command-line) """
        for key in args_dict:
            if args_dict[key] is None: continue
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

        """ Dictionary for pde given the dataset used """
        self.pde = {
        "laplace1D_cos": "laplace",
        "laplace2D_cos": "laplace"
        }

        """ Dictionary for I/O dimensions of the dataset used """
        self.phys_dom = {
        "laplace1D_cos": (1,1,1),
        "laplace2D_cos": (2,1,1)
        }

        """ Dictionary for I/O dimensions of the network used """
        self.comp_dom = {
        "laplace1D_cos": (1,1,1),
        "laplace2D_cos": (2,1,1)
        }
