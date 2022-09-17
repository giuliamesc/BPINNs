import tensorflow as tf

"""


"""

def tf_unpack(tensor):
    return tf.unstack(tf.expand_dims(tensor, axis=-2), axis=-1) 

def tf_trace(lt):
    return tf.expand_dims(sum([v[:,i] for i, v in enumerate(lt)]), axis=-1)


def gradient_scalar(tape, s, x):
    """
    input  shape: (n_sample x 1)
    output shape: (n_sample x dim_inp)
    """
    return tape.gradient(s, x)

def divergence_vector(tape, r, x):
    """
    input  shape: (n_sample x dim_inp)
    output shape: (n_sample x 1)
    """
    c = tf_unpack(r)
    return tf_trace(gradient_vector(tape, c, x))

def laplacian_scalar(tape, s, x):
    """
    input  shape: (n_sample x 1)
    output shape: (n_sample x 1)
    """
    gs = gradient_scalar(tape,s,x) 
    return divergence_vector(tape, gs, x)



def gradient_vector(tape, c, x):
    """
    input  shape: (n_sample x 1      ) x dim_out
    output shape: (n_sample x dim_inp) x dim_out
    """
    return  [gradient_scalar(tape, gs, x) for gs in c]

def divergence_tensor(tape, t, x):
    """
    input  shape: (n_sample x dim_inp) x dim_out
    output shape: (n_sample x 1      ) x dim_out
    """
    return [divergence_vector(tape, dv, x) for dv in t]

def laplacian_vector(tape, c, x):
    """
    input  shape: (n_sample x 1) x dim_out
    output shape: (n_sample x 1) x dim_out 
    """
    return [laplacian_scalar(tape, ls, x) for ls in c]