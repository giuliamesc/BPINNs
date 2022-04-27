from argparse import ArgumentParser

"""
Class to handle all the command-line arguments:
the first [--method] is mandatory (can be SVGD or HMC) and specify the BayesianNN method to use;
the second [--config] is a json file (in the subfolder "config") where we can find all the parameters (default is "defaul.json")

You can also overspecified all the parameters in the json file here, directly from command-line.
"""
class Parser(ArgumentParser):
    def __init__(self):
        """Initializer"""
        super(Parser, self).__init__(description='Bayesian PINN for Inverse Eikonal')

        self.add_argument('--method', type=str, default="HMC", help="""Methods to use for BPINN. Available:
        															- HMC  (Hamiltonian Monte Carlo)
        															- SVGD (Stein Variational Gradient Descent)""")

        self.add_argument('--config', type=str, default="default", help="""name of json file where we can find all the parameters. 
        																	You have to provide the parameters for at least the method you've selected.
        																	You can also overwrite some parameters directly from terminal
        																	""")


        # Architecture
        self.add_argument('--n_layers' , type=int, help='Need >=1, number of hidden layers in the NN')
        self.add_argument('--n_neurons', type=int, help='Need >=1, number of neurons in each hidden layer in the NN')

        # Experiment
        self.add_argument('--dataset', type=str, help="""Choose the experiment :
                                                        - laplace1D_cos (1D, Laplace)
                                                        - laplace2D_cos (2D, Laplace)
                                                        """)
        self.add_argument('--num_collocation', type=int,   help="Need to be integer. Number of Domain Data to use as collocation data")
        self.add_argument('--num_fitting',     type=int,   help="Need to be integer. Number of Domain Data to use as sparse Exact data")
        self.add_argument('--noise_lv',        type=float, help="Artificial noise in exact dataset")
        self.add_argument('--batch_size',      type=int,   help="Batch size for training collocation. Select 0 if you don't want a batch")

        # Param
        self.add_argument('--param_res'  , type=float, help="weight for pde log loss")
        self.add_argument('--param_data' , type=float, help="weight for data log loss")
        self.add_argument('--param_prior', type=float, help="weight for prior log loss")

        # Sigmas
        self.add_argument('--data_prior_noise', type=float, help='noise in data prior (sigma_D)^2')
        self.add_argument('--pde_prior_noise' , type=float, help='noise in pde prior (sigma_R)^2')
        self.add_argument('--data_prior_noise_trainable', type=bool, help='Train on (sigma_D)^2 as a hyperparameter')
        self.add_argument('--pde_prior_noise_trainable' , type=bool, help='Train on (sigma_R)^2 as a hyperparameter')

		# HMC
        self.add_argument('--N_HMC', type=int, help="N: number of samples in HMC")
        self.add_argument('--M_HMC', type=int, help="M: number of samples to use in HMC (after burnin). Need <= N")
        self.add_argument('--L_HMC', type=int, help="L: number of leapfrog step in HMC")
        self.add_argument('--dt_HMC', type=float, help="dt: step size in HMC")
        self.add_argument('--dt_noise_HMC', type=float, help="dt_noise: step size in HMC for log betas")

        # SVGD
        self.add_argument('--n_samples', type=int,   help='(5-30) number of model instances in SVGD')   # number of NNs used
        self.add_argument('--epochs'   , type=int,   help='number of epochs to train')
        self.add_argument('--lr'       , type=float, help='learning rate for NN parameters')
        self.add_argument('--lr_noise' , type=float, help='learnign rate for log beta, if trainable')
        self.add_argument('--param_repulsivity', type=float, help='parameter for repulsivity in SVGD')

        # Utils
        self.add_argument('--random_seed', type=int,  help="random seed for numpy and tf random generator")
        self.add_argument('--debug_flag' , type=bool, help='prints loss value, h, h0, h1 and general debug utilities at each iteration')
        self.add_argument('--save_flag'  , type=bool, help='flag for save results in a new folder')
        