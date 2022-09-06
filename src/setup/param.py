class Param:
    """Initializer"""
    def __init__(self, hp, args):

        self.problem   = hp["general"]["problem"]   # problem used
        self.case_name = hp["general"]["case_name"] # case used
        self.method    = hp["general"]["method"]    # method used: SVGD, HMC, VI, ...
        if self.case_name == "": self.case_name = "default"

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

        # Convert string of boolean in boolean
        self.__change_string_to_bool()

    @property
    def data_config(self):
        return self.config

    @data_config.setter
    def data_config(self, data_config): 
        
        self.config  = data_config
        self.pde     = data_config.pde
        self.physics = data_config.physics
        self.folder_name = data_config.name
        # set all useful parameters from physical domain and dimension
        self.phys_dim = Dimension(data_config, True)
        self.comp_dim = Dimension(data_config, False)

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
        self.utils["gen_flag"]   = self.__string_to_bool(self.utils["gen_flag"])

    def __command_line_update_params(self, args_dict):
        """ Update the parameter given by json file using args (overspecification by command-line) """
        for key in args_dict:
            if args_dict[key] is None: continue
            if key == "problem"   : self.problem   = args_dict[key]
            if key == "case_name" : self.case_name = args_dict[key]
            if key == "method"    : self.method    = args_dict[key]
            for jdict in [self.experiment, self.architecture, self.coeff, self.sigmas, self.utils]:
                if key in jdict: jdict[key] = args_dict[key]

    def __command_line_update_method(self, args_dict):
        """ Update the parameter given by json file using args (overspecification by command-line) """
        for key in args_dict:
            if args_dict[key] is None: continue
            if key in self.param_method:
                self.param_method[key] = args_dict[key]


class Dimension():
    
    def __init__(self, data_config, physical):

        self.dimensions = data_config.phys_dom if physical else data_config.comp_dom

        ## store the dimension input (1D, 2D or 3D)
        self.n_input   = self.dimensions["n_input"]
        ## dimension of solution
        self.n_out_sol = self.dimensions["n_out_sol"]
        ## dimension of parametric field
        self.n_out_par = self.dimensions["n_out_par"]
