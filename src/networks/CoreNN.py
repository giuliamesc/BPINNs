import tensorflow as tf

class CoreNN():

    """
    - Builds initial network
    - Contains layers and weights and biases (theta)
    - Does forward pass on given theta
    - Can re-sample theta given a seed
    
    Neural networks parameters (theta)
        - nn_params = (weights, biases)
        - weights is a list of numpy arrays 
            - 1 layer               : shape (n_input, n_neurons)
            - (n_layers - 1) layers : shape (n_neurons, n_neurons) 
            - 1 layer               : shape (n_neurons, n_out_sol)
        - biases is a list of numpy arrays 
            - n_layers layers       : shape (n_neurons,) 
            - 1 layer               : shape (n_out_sol,)
    """

    def __init__(self, par):

        # Domain dimensions
        self.n_inputs  = par.comp_dim.n_input
        self.n_out_sol = par.comp_dim.n_out_sol

        # Architecture parameters
        self.n_layers   = par.architecture["n_layers"]
        self.n_neurons  = par.architecture["n_neurons"]
        self.activation = par.architecture["activation"]

        # Build the Neural network architecture
        self.model = self.__build_NN(par.utils["random_seed"])
        self.dim_theta = self.__compute_dim_theta()
        
    @property
    def nn_params(self):
        """ Getter for nn_params property """
        weights = [layer.get_weights()[0] for layer in self.model.layers]
        biases  = [layer.get_weights()[1] for layer in self.model.layers]
        theta = list()
        for w, b in zip(weights, biases):
            theta.append(w)
            theta.append(b)
        return theta

    @nn_params.setter
    def nn_params(self, theta):
        """ Setter for nn_params property """
        for layer, weight, bias in zip(self.model.layers, theta[0::2], theta[1::2]):
            layer.set_weights((weight,bias))

    def __build_NN(self, seed):
        """
        Initializes a fully connected Neural Network with 
        - Glorot Uniform initialization of weights
        - Zero initialization for biases
        """
        # Set random seed for inizialization
        tf.random.set_seed(seed)
        # Input Layer
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(self.n_inputs,)))
        # Hidden Layers
        for _ in range(self.n_layers):
            model.add(tf.keras.layers.Dense(self.n_neurons, activation=self.activation, 
                      kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        # Output Layer
        model.add(tf.keras.layers.Dense(self.n_out_sol, 
                      kernel_initializer='glorot_uniform', bias_initializer='zeros'))

        return model

    def initialize_NN(self, seed):
        """ Initialization of the Neural Network with given random seed """
        self.model = self.__build_NN(seed)
        
    def __compute_dim_theta(self):
        return sum([tf.size(wb).numpy() for wb in self.nn_params])

    def forward(self, inputs):
        """ 
        Simple prediction on draft of Solution
        inputs : np array  (n_samples, n_input)
        output : tf tensor (n_samples, n_out_sol)
        """
        x = tf.convert_to_tensor(inputs)
        return self.model(x)
