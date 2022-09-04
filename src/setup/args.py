from argparse import ArgumentParser

class Parser(ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Bayesian PINN for PDEs')

        # Configuration (choose configuration file to set other parameters)
        self.add_argument('--config', type=str, default="default", 
                        help="""Name of json file where we can find all the parameters. 
    							You have to provide the parameters for at least the method you've selected.
        						You can also overwrite some parameters directly from terminal
                            """)
        # Problem (choose the physical problem)
        self.add_argument('--problem', type=str, 
                        help="""Choose the experiment :
                                - laplace1D (1D, Laplace)
                                - laplace2D (2D, Laplace)
                            """)
        # Dataset (choose the data for the problem)
        self.add_argument('--case_name', type=str, 
                        help="""Choose the experiment :
                                - cos (1D-2D, Laplace)
                            """)
        # Algorithm (choose training algorithm)
        self.add_argument('--method', type=str, 
                        help="""Methods to use for BPINN. Available:
        		                - HMC  (Hamiltonian Monte Carlo)
        			            - SVGD (Stein Variational Gradient Descent)
                                - VI   (variational inference)
                                - TEST (use for debug purpouses)
                            """)

        # Experiment
        self.add_argument('--num_collocation', type=int,   help="Needs to be integer. Number of Domain Data to use as collocation data")
        self.add_argument('--num_fitting',     type=int,   help="Needs to be integer. Number of Domain Data to use as sparse exact data")
        self.add_argument('--noise_lv',        type=float, help="Artificial noise in exact dataset")
        self.add_argument('--batch_size',      type=int,   help="Batch size for training collocation. Select 0 if you don't want a batch")

        # Architecture
        self.add_argument('--activation', type=str, help='Activation function for hidden layers')
        self.add_argument('--n_layers'  , type=int, help='Number of hidden layers in the NN')
        self.add_argument('--n_neurons' , type=int, help='Number of neurons in each hidden layer in the NN')

        # Coefficients
        self.add_argument('--res'  , type=float, help='Weight for   pde log loss')
        self.add_argument('--data' , type=float, help='Weight for  data log loss')
        self.add_argument('--prior', type=float, help='Weight for prior log loss')

        # Sigmas (prior noise)
        self.add_argument('--data_pn', type=float, help='Noise in data prior (sigma_D)^2')
        self.add_argument( '--pde_pn', type=float, help='Noise in  pde prior (sigma_R)^2')
        self.add_argument('--data_pn_flag', type=bool, help='Train on (sigma_D)^2 as a hyperparameter')
        self.add_argument( '--pde_pn_flag', type=bool, help='Train on (sigma_R)^2 as a hyperparameter')

        # Utils
        self.add_argument('--random_seed', type=int,  help='Random seed for np and tf random generator')
        self.add_argument('--debug_flag' , type=bool, help='Prints general debug utilities at each iteration')
        self.add_argument('--save_flag'  , type=bool, help='Flag to save results in a new folder')
        self.add_argument('--gen_flag'   , type=bool, help='Flag for new data generation')

        # %% Algoritm Parameters
        self.add_argument('--epochs', type=int, help='Number of epochs to train')

        # HMC
        self.add_argument('--burn-in', type=int,   help="Number of samples to use in HMC (after burn-in). Needs to be <= epochs")
        self.add_argument('--HMC_L'  , type=int,   help="L: number of leapfrog step in HMC")
        self.add_argument('--HMC_dt' , type=float, help="dt: step size in HMC")
        self.add_argument('--HMC_ns' , type=float, help="ns: step size in HMC for log betas")

        # SVGD

        # VI


