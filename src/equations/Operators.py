import tensorflow as tf

class Operators:
    
    """
    Class for differential operators
    
    Notations:
    x is the input variable (n_sample x dim_inp)
    s is a scalar     (n_sample x 1)
    r is a row vector (n_sample x dim_inp)
    c is a col vector (n_sample x 1) x dim_out
    t is a tensor     (n_sample x dim_inp) x dim_out
    """

    @staticmethod
    def tf_convert(tensor): 
        """ Conversion of a numpy array to tensor """
        return tf.cast(tensor, dtype=tf.float32)

    @staticmethod
    def tf_unpack(tensor):
        """ Returns a list whose elements are the tensor representing the columns of the input tensor """
        return tf.unstack(tf.expand_dims(tensor, axis=-2), axis=-1) 

    @staticmethod
    def tf_pack(tensor_list):
        """ Returns a list whose elements are the tensor representing the columns of the input tensor """
        return tf.squeeze(tf.stack(tensor_list, axis=-1), axis=-2)

    @staticmethod
    def tf_trace(lt):
        """ Computes the trace (in the algebrical sense), but for an input which is a list of column tensors """
        return tf.expand_dims(sum([v[:,i] for i, v in enumerate(lt)]), axis=-1)

    @staticmethod
    def gradient_scalar(tape, s, x):
        """
        input  shape: (n_sample x 1)
        output shape: (n_sample x dim_inp)
        """
        return tape.gradient(s, x)
    
    @staticmethod    
    def divergence_vector(tape, r, x):
        """
        input  shape: (n_sample x dim_inp)
        output shape: (n_sample x 1)
        """
        c = Operators.tf_unpack(r)
        return Operators.tf_trace(Operators.gradient_vector(tape, c, x))
    
    @staticmethod
    def laplacian_scalar(tape, s, x):
        """
        input  shape: (n_sample x 1)
        output shape: (n_sample x 1)
        """
        gs = Operators.gradient_scalar(tape,s,x) 
        return Operators.divergence_vector(tape, gs, x)
        
    @staticmethod    
    def gradient_vector(tape, c, x):
        """
        input  shape: (n_sample x 1      ) x dim_out
        output shape: (n_sample x dim_inp) x dim_out
        """
        return  [Operators.gradient_scalar(tape, gs, x) for gs in c]
        
    @staticmethod
    def divergence_tensor(tape, t, x):
        """
        input  shape: (n_sample x dim_inp) x dim_out
        output shape: (n_sample x 1      ) x dim_out
        """
        return [Operators.divergence_vector(tape, dv, x) for dv in t]
    
    @staticmethod
    def laplacian_vector(tape, c, x):
        """
        input  shape: (n_sample x 1) x dim_out
        output shape: (n_sample x 1) x dim_out 
        """
        return [Operators.laplacian_scalar(tape, ls, x) for ls in c]