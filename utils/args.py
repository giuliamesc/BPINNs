import argparse

"""
Class to handle all the command-line arguments:
the first [--method] is mandatory (can be SVGD or HMC) and specify the BayesianNN method to use;
the second [--config] is a json file (in the subfolder "config") where we can find all the parameters (default is "default.json")

You can also overspecified all the parameters in the json file here, directly from command-line.
"""
class Parser(argparse.ArgumentParser):
    def __init__(self):
        """Initializer"""
        super(Parser, self).__init__(description='Bayesian PINN for Inverse Eikonal')

        self.add_argument('--method', type=str, default="SVGD", required=True, help="""Methods to use for BPINN. Available:
        																			  -HMC (Hamiltonian Monte Carlo -> MCMC method)
        																			  -SVGD (Stein Variational Gradient Descend -> Finite opt. method)""")

        self.add_argument('--config', type=str, default="default.json", help="""json file where we can find all the parameters. In default.json you can find an example.
        																		You have to provide the parameters for at least the method you've selected.
        																		You can also overwrite some parameters directly from terminal using the additional specification below
        																		(this can be useful for change only a single param during experiments, without modifing the entire json file)""")

		### Additional parameters overspecified
        # architecture
        self.add_argument('--n_layers', type=int, help='Need >=1, number of hidden layers in the NN')
        self.add_argument('--n_neurons', type=int, help='Need >=1, number of neurons in each hidden layer in the NN')

        # experiment
        self.add_argument('--dataset', type=str, help="""Choose the experiment :
                                                        - exponential (1D, Isotropic)
                                                        - circle (2D, Isotropic)
                                                        - triflag (2D, Isotropic)
                                                        - checkerboard (2D, Isotropic)
                                                        - square_with_circle (2D, Isotropic)
                                                        - anisotropic1 (2D, Anisotropic)
                                                        - anisotropic2 (2D, Anisotropic)
                                                        - cube3D (3D, Isotropic)
                                                        - prolate3D (3D, Isotropic)
                                                        - prolate3D_scar (3D, Isotropic)""")

        self.add_argument('--prop_exact', type=float, help="Need to be between 0 and 1. Proportion of Domain Data to use as Sparse Exact data")
        self.add_argument('--prop_collocation', type=float, help="Need to be between 0 and 1. Proportion of Domain Data to use as collocation data")
        self.add_argument('--noise_lv', type=float, help="noise in exact dataset")
        self.add_argument('--is_uniform_exact', type=bool, help="Flag for uniform grid exact data")
        self.add_argument('--batch_size', type=int, help="""batch size for training collocation.
                                                            Need to be <= (Num of Domain)*prop_collocation.
                                                            Select 0 if you don't want a """)

        # param
        self.add_argument('--param_pde', type=float, help="weight for eikonal log loss")
        self.add_argument('--param_data', type=float, help="weight for data log loss")
        self.add_argument('--param_prior', type=float, help="weight for prio log loss")
        self.add_argument('--random_seed', type=int, help="random seed for numpy and tf random generator")

        # sigmas
        self.add_argument('--data_prior_noise', type=float, help='noise in data prior (sigma_D)^2')
        self.add_argument('--pde_prior_noise', type=float, help='noise in pde prior (sigma_R)^2')
        self.add_argument('--data_prior_noise_trainable', type=bool, help='Train on (sigma_D)^2 as a hyperparameter')
        self.add_argument('--pde_prior_noise_trainable', type=bool, help='Train on (sigma_R)^2 as a hyperparameter')

        # SVGD
        self.add_argument('--n_samples', type=int, help='(5-30) number of model instances in SVGD')   # number of NNs used
        self.add_argument('--epochs', type=int, help='number of epochs to train')
        self.add_argument('--lr', type=float, help='learning rate for NN parameters')
        self.add_argument('--lr_noise', type=float, help='learnign rate for log beta, if trainable')
        self.add_argument('--param_repulsivity', type=float, help='parameter for repulsivity in SVGD')

		# HMC
        self.add_argument('--N_HMC', type=int, help="N: number of samples in HMC")
        self.add_argument('--M_HMC', type=int, help="M: number of samples to use in HMC (after burnin). Need <= N")
        self.add_argument('--L_HMC', type=int, help="L: number of leapfrog step in HMC")
        self.add_argument('--dt_HMC', type=float, help="dt: step size in HMC")
        self.add_argument('--dt_noise_HMC', type=float, help="dt_noise: step size in HMC for log betas")

        # utilities
        self.add_argument('--verbose', type=bool, help='print all the info at every epoch')
        self.add_argument('--save_flag', type=bool, help='flag for save results in a new folder (or send it in "trash" folder)')


    def parse(self):
        """Parser"""
        args = self.parse_args()
        return args

# global
args = Parser().parse()
