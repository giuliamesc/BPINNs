import tensorflow as tf

class Operators:
    
    """
    Class for differential operators
    
    Notations:
    x is the derivation variable (n_sample x dim_input)
    d is dim_output
    s is a scalar function (n_sample x 1)
    v is a vector function (n_sample x dim_output)
    A is a tensor function (n_sample x dim_output x dim_output)
    """
    
    @staticmethod
    def gradient_scalar(tape, s, x):
        """
        input  shape: (n_sample x 1)
        output shape: (n_sample x 1 x dim_input)
        """
        return tf.expand_dims(tape.gradient(s, x), axis=1)

    @staticmethod
    def gradient_vector(tape, v, x, d):
        """
        input  shape: (n_sample x dim_output)
        output shape: (n_sample x dim_output x dim_input)
        """
        return tf.stack([tape.gradient(v[:,i], x) for i in range(d)], axis=-2)
    

    @staticmethod
    def derivate_scalar(tape, s, x):
        """
        input  shape: (n_sample x 1)
        output shape: (n_sample x 1)
        """
        return tf.squeeze(Operators.gradient_scalar(tape, s, x), axis=-1)

    @staticmethod
    def derivate_vector(tape, s, x, d):
        """
        input  shape: (n_sample x dim_output)
        output shape: (n_sample x dim_output)
        """
        return tf.squeeze(Operators.gradient_vector(tape, s, x, d), axis=-1)


    @staticmethod
    def divergence_vector(tape, v, x, d):
        """
        input  shape: (n_sample x dim_output) or
        input  shape: (n_sample x 1 x dim_input)
        output shape: (n_sample x 1)
        """
        if len(v.shape) == 3: v = tf.squeeze(v, axis = -2)
        return tf.expand_dims(tf.linalg.trace(Operators.gradient_vector(tape, v, x, d)), axis=-1)

    @staticmethod
    def divergence_tensor(tape, A, x, d):
        """
        input  shape: (n_sample x dim_output x dim_input)
        output shape: (n_sample x dim_output)
        """
        divergences = [Operators.divergence_vector(tape, A[:,i,:], x, d) for i in range(d)]
        return tf.squeeze(tf.stack(divergences, axis = -1), axis=-2)


    @staticmethod
    def laplacian_scalar(tape, s, x, d):
        """
        input  shape: (n_sample x 1)
        output shape: (n_sample x 1)
        """
        return Operators.divergence_vector(tape, Operators.gradient_scalar(tape, s, x), x, d)

    @staticmethod
    def laplacian_vector(tape, v, x, d):
        """
        input  shape: (n_sample x dim_output)
        output shape: (n_sample x dim_output)
        """
        return Operators.divergence_tensor(tape, Operators.gradient_vector(tape, v, x, d), x, d)
