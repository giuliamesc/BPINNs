import tensorflow as tf

class operators:
    
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
        return tf.stack([tape.gradient(v[:,i], x) for i in range(d)], axis = -2)
    
    @staticmethod
    def derivate_scalar(tape, s, x):
        """
        input  shape: (n_sample x 1)
        output shape: (n_sample x 1)
        """
        return tf.squeeze(operators.gradient_scalar(tape, s, x), axis =-1)
    @staticmethod
    def derivate_vector(tape, s, x, d):
        """
        input  shape: (n_sample x dim_output)
        output shape: (n_sample x dim_output)
        """
        return tf.squeeze(operators.gradient_vector(tape, s, x, d), axis =-1)

    @staticmethod
    def divergence_vector(tape, v, x, d):
        """
        input  shape: (n_sample x dim_output) or
        input  shape: (n_sample x 1 x dim_input)
        output shape: (n_sample x 1)
        """
        if len(v.shape) == 3: v = tf.squeeze(v, axis = -2)
        return tf.expand_dims(tf.linalg.trace(operators.gradient_vector(tape, v, x, d)), axis = -1)
    @staticmethod
    def divergence_tensor(tape, A, x, d):
        """
        input  shape: (n_sample x dim_output x dim_input)
        output shape: (n_sample x dim_output)
        """
        divergences = [operators.divergence_vector(tape, A[:,i,:], x, d) for i in range(d)]
        return tf.squeeze(tf.stack(divergences, axis = -1), axis = -2)

    @staticmethod
    def laplacian_scalar(tape, s, x, d):
        """
        input  shape: (n_sample x 1)
        output shape: (n_sample x 1)
        """
        return operators.divergence_vector(tape, operators.gradient_scalar(tape, s, x), x, d)
    @staticmethod
    def laplacian_vector(tape, v, x, d):
        """
        input  shape: (n_sample x dim_output)
        output shape: (n_sample x dim_output)
        """
        return operators.divergence_tensor(tape, operators.gradient_vector(tape, v, x, d), x, d)


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
    

class laplace(pde_constraint):
    def __init__(self, par):

        super().__init__(par.n_input, par.n_out_sol, par.n_out_par)
        
    def compute_pde_residual(self, x, u, f):
        op = operators()
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            lap = op.laplacian_vector(tape, u, x, self.n_out_sol)
        return lap + f

    def compute_pde_losses(self, x, u, f):
        """
        - Laplacian(u) = f -> f + Laplacian(u) = 0
        u shape: (n_sample x n_out_sol)
        f shape: (n_sample x n_out_par)    
        """
        residual = compute_pde_residual(x, u, f)
        return residual


if __name__ == "__main__":
    ns = 4
    di = 3
    do = 1
    x  = tf.sort(tf.Variable(tf.random.normal((ns, di))), axis=0)

    op = operators()
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        us = tf.sin(x[:,0:1])
        uv = tf.stack([us[:,0]]*do, axis = -1)
        ut = tf.stack([uv]*di, axis = -1)
        
        us_x = op.gradient_scalar(tape, us, x)
        uv_x = op.gradient_vector(tape, uv, x, do)
        
        if di == 1:
            us_d = op.derivate_scalar(tape, us, x)
            uv_d = op.derivate_vector(tape, uv, x, do)
        
        print("us:   ", us.shape)
        print("us_x: ", us_x.shape)
        if di == 1: 
            print("us_d: ", us_d.shape)
        
        
        print("uv:   ", uv.shape)
        print("uv_x: ", uv_x.shape)
        if di == 1: 
            print("uv_d: ", uv_d.shape)
        
        
        print("div_v:", op.divergence_vector(tape,uv,x,do).shape)
        print("div_v:", op.divergence_vector(tape,us_x,x,do).shape)
        
        print("div_A:", op.divergence_tensor(tape,ut,x,do).shape)
        
        print("Lap_s:", op.laplacian_scalar(tape, us, x, do).shape)
        print("Lap_v:", op.laplacian_vector(tape, uv, x, do).shape)
        


    
    