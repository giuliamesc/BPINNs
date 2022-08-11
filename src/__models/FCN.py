import tensorflow as tf
from keras import layers


class Net:
    """!
    Define the base neural network model: Fully Connected Network

    Model defined using tf.keras.Sequential()
    The final architecture will be:
    [n_input, n_hidden_neuron, n_hidden_neuron,...,n_hidden_neuron, n_output]
    with n_hidden_neuron repated for n_hidden_layer times

    @param n_input input dimension (1, 2 or 3)
    @param n_hidden_layer number of hidden layers
    @param n_hidden_neuron number of neurons for each hidden layer
    @param n_output dimension of output = n_out_sol+n_out_par
    """
    def __init__(self, n_input, n_hidden_layer, n_hidden_neuron, n_output):
        """Constructor: build our neural network architecture"""

        ## features: our neural network model build with tf.keras.Sequential()
        self._features = tf.keras.Sequential()   # use sequential to build the model

        self._features.add(tf.keras.Input(shape=(n_input,)))    #input

        for i in range(n_hidden_layer): #for loop to add the n_hidden_layers (with n_hidden_neuron each)
            self._features.add(layers.Dense(n_hidden_neuron, activation="swish", kernel_initializer='glorot_uniform', bias_initializer='zeros'))

        self._features.add(layers.Dense(n_output,  kernel_initializer='glorot_uniform', bias_initializer='zeros'))   #output


    # return the number of parameters in the network
    def num_parameters(self):
        """return the number of parameters in the network"""
        tot = 0
        for layer in self._features.layers:
            (dim1,dim2) = layer.get_weights()[0].shape
            (dim3,) = layer.get_weights()[1].shape
            tot+= dim1*dim2
            tot+= dim3
        return tot

    # return a list of all dimension of W and b in each layer
    def get_dimensions(self):
        """return a list of all dimension of W and b in each layer"""
        architecture = []
        for layer in self._features.layers:
            (dim1,dim2) = layer.get_weights()[0].shape # dims of matrix W
            (dim3,) = layer.get_weights()[1].shape  # dim of vector bias
            architecture.append((dim1,dim2,dim3))
        return architecture


    def forward(self, x):
        """!
        forward pass of the input x
        @param x input
        """
        return self._features(x)

    # return a list of all the matrix W and bias vectors in the network (e.g. shape=(8,))
    def get_parameters(self):
        """ Get the trainable parameters of the NN """
        return self._features.trainable_weights

    def get_weights(self):
        """ Get the parameters of the NN in numpy tensors """
        list_par = []
        for layer in self._features.layers:
            list_par.append( layer.get_weights()[0] )
            list_par.append( layer.get_weights()[1] )
        return list_par

    def update_weights(self, param):
        """!
        Update the weights giving a list of tensors
        @param param list of tf tensor represent the new weights for the network
        """
        i = 0
        for layer in self._features.layers:
            # use the method set_weights for each layer
            # first param is the matrix W, second the bias vector b
            layer.set_weights((param[i], param[i+1]))
            i+=2
