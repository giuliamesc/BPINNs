import numpy as np
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_probability as tfp

class BayesNN():
    """
    - Contain the layers and weights
    - Can do forward and predict
    """
    
    def __init__(self, par):
        
        # Domain dimensions
        self.n_inputs  = par.n_inputs
        self.n_out_sol = par.n_out_sol
        self.n_out_par = par.n_out_par

        # Architecture parameters
        self.n_layers   = par.architecture["n_layers"]
        self.n_neurons  = par.architecture["n_neurons"]
        self.activation = par.architecture["activation"]

        # Build the Neural network architecture
        self.model = self.__build_NN()

    def __build_NN(self):
        """
        Initializes a fully connected Neural Network with Glorot Uniform initialization of weights and biases initialized to zero.
        - model is an initialized instance of neural network
        """
        model = tf.keras.Sequential()

        # Input Layer
        model.add(tf.keras.Input(shape=(self.n_inputs,)))
        
        # Hidden Layers
        for _ in range(self.n_layers):
            model.add(tf.keras.layers.Dense(self.n_neurons, activation=self.activation, 
                      kernel_initializer='glorot_uniform', bias_initializer='zeros'))

        # Output Layer
        model.add(tf.keras.layers.Dense(self.n_out_sol+self.n_out_par, 
                  kernel_initializer='glorot_uniform', bias_initializer='zeros'))

        return model
        
    @property
    def nn_params(self):
        """
        Returns the params of the nn: 
        - nn_params = (weights, biases)
        - weights is a list of numpy arrays 
            - 1 layer               : shape (n_input, n_neurons)
            - (n_layers - 1) layers : shape (n_neurons, n_neurons) 
            - 1 layer               : shape (n_neurons, n_out_sol+n_out_par)
        - biases is a list of numpy arrays 
            - n_layers layers       : shape (n_neurons,) 
            - 1 layer               : shape (n_out_sol+n_out_par,)
        """
        weights = [layer.get_weights()[0] for layer in self.model.layers]
        biases  = [layer.get_weights()[1] for layer in self.model.layers]
        return (weights, biases)

    @nn_params.setter
    def nn_params(self, new_params):
        """
        Sets the params of the nn: 
        - new_params = (weights, biases)
        - weights is a list of numpy arrays 
            - 1 layer               : shape (n_input, n_neurons)
            - (n_layers - 1) layers : shape (n_neurons, n_neurons) 
            - 1 layer               : shape (n_neurons, n_out_sol+n_out_par)
        - biases is a list of numpy arrays 
            - n_layers layers       : shape (n_neurons,) 
            - 1 layer               : shape (n_out_sol+n_out_par,)
        """
        weights, biases = new_params
        for layer, weight, bias in zip(self.model.layers, weights, biases):
            layer.set_weights((weight,bias))

    def sample(self):
        pass

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.forward(x)

# ONLY FOR DEBUG
class TempPar:
    def __init__(self):
        
        self.n_inputs  = 1
        self.n_out_sol = 1
        self.n_out_par = 1

        self.architecture = {
            "activation": "swish",
            "n_layers" : 3,
            "n_neurons": 5
        }

def create_weights(par):

    n_neurons = par.architecture["n_neurons"]
    n_layers  = par.architecture["n_layers"]

    weights = [np.random.randn(par.n_inputs, n_neurons)]
    biases = [np.random.randn(n_neurons)]

    for _ in range(n_layers-1):
        weights.append(np.random.randn(n_neurons, n_neurons))
        biases.append(np.random.randn(n_neurons))

    weights.append(np.random.randn(n_neurons, par.n_out_sol+par.n_out_par))
    biases.append(np.random.randn(par.n_out_sol+par.n_out_par))

    return (weights, biases)

if __name__ == "__main__":
    
    par = TempPar()
    n_sample = 4
    bayes_nn = BayesNN(par)

    x = tf.random.uniform(shape=[n_sample, bayes_nn.n_inputs])
    bayes_nn.nn_params = create_weights(par)
    print(bayes_nn.nn_params)