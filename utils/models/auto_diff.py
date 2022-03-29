import tensorflow as tf

class pde_constraint:
    """
    Class for the pde_constraint
    """

    def __init__(self, n_input, n_out_sol, n_out_par):
        """! Constructor
        @param n_input dimension input (1,2 or 3)
        @param n_out_sol dimension of solution
        @param n_out_par dimension of parametric field
        """
        
        self.n_input = n_input
        self.n_out_sol = n_out_sol
        self.n_out_par = n_out_par

    def compute_pde_losses(self, u_gr, f_gr, f): ## MIGLIORA CON ABC
        """compute the pde losses, need to be override in child classes"""
        return 0.
    
class operators:
    """
    Class for differential operators
    
    Notations
    d is the dimension of the space
    x is the derivation variable (n_sample x 1)
    s is a scalar function (n_sample x 1)
    v is a vector function (n_sample x d)
    A is a tensor function (n_sample x d x d)
    """
    @staticmethod
    def gradient_scalar(tape, s, x):
        return tape.gradient(s, x)

    @staticmethod
    def gradient_vector(tape, v, x, d):
        # d = int(tf.shape(x)[-1])
        return  tf.stack([tape.gradient(v[:,i], x) for i in range(d)], axis = -2)

    @staticmethod
    def divergence_vector(tape, v, x, d):
        # d = int(tf.shape(x)[-1])
        # return sum([tape.gradient(v[:,i], x)[:,i] for i in range(d)])
        return tf.linalg.trace(tens_style.gradient_vector(tape, v, x, d))

    @staticmethod
    def divergence_tensor(tape, A, x, d):
        # d = int(tf.shape(x)[-1])
        return tf.stack([tens_style.divergence_vector(tape, A[:,i,:], x, d) for i in range(d)], axis = -1)

    @staticmethod
    def laplacian_scalar(tape, s, x, d):
        return operators.divergence_vector(tape, tape.gradient(s, x), x, d)

    @staticmethod
    def laplacian_vector(tape, v, x, d):
        return operators.divergence_tensor(tape, tens_style.gradient_vector(tape, v, x, d), x, d)

    

class laplace(pde_constraint):
    def __init__(self, par):

        super().__init__(par.n_input, par.n_out_sol, par.n_out_par)

    def compute_pde_losses(self, u_gr_2, f):
        """
        - Laplacian(u) = f -> f + Laplacian(u) = 0
        """
        # compute the loss 1: residual of the PDE

        loss_1 = tf.expand_dims(tf.reduce_sum(f, axis=-1),axis=-1)

        for u_gr_2_comp in u_gr_2:
            loss_1 += u_gr_2_comp
        
        return loss_1


if __name__ == "__main__":
    w1 = tf.Variable(tf.random.normal((3, 2)), name='w1')
    b1 = tf.Variable(tf.random.normal((2)), name='b1')
    w2 = tf.Variable(tf.random.normal((2, 2)), name='w2')
    b2 = tf.Variable(tf.random.normal((2)), name='b2')
    x = [[1., 2., 3.]]

    with tf.GradientTape(persistent=True) as tape:
        y1 = x @ w + b
        y2 = x @ w + b
        loss = tf.reduce_mean(y2**2)
    print(y)