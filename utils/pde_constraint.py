import tensorflow as tf


class pde_constraint:
    """
    Class for the pde_constraint (already implemented: eikonal and anisotropic eikonal)
    (in working-progress diffusion-eikonal)

    Parent class used for inheritance of eikonal and anisotropic_eikonal
    """

    def __init__(self, bayes_methods, n_input, n_output_vel):
        """! Constructor
        @param bayes_methods name of method used
        @param n_input dimension input (1,2 or 3)
        @param n_output_vel dimension of velocity output (1 isotropic, >1 anisotropic)"""
        self.bayes_methods = bayes_methods # "SVGD", "HMC", ...
        self.n_input = n_input  # 1,2,3 (1D, 2D, 3D)
        self.n_output_vel = n_output_vel # 1 for isotropic, >1 for anisotropic


    def compute_pde_losses(self, at_gr, v_gr, v):
        """compute the pde losses, need to be override in child classes"""
        return 0.


class eikonal(pde_constraint):
    """
    Child class of pde_constraint

    Implement the isotropic eikonal
    loss1 => isotropic eikonal ||Grad at|| = 1/v
    loss2 => penalizing high gradients of v, ||Grad v|| small enough
    """

    # Isotropic case
    def __init__(self, par):
        """! Constructor
        @param par an object of our param class
        """

        """Use the parent constructor (pde_constraint.__init__)"""
        super().__init__(par.method, par.n_input, par.n_output_vel)

        # store the param2loss for loss_2
        self.param2loss = par.param["param2loss"]

    def compute_pde_losses(self, at_gr, v_gr, v):
        """compute the pde losses (override parent method)
        @param at_gr list of tensors (at_x, at_y, at_z) (if 3D), where at_x is the derivative of at wrt x
        @param v_gr list of tensors (v_x, v_y, v_z) (if 3D), where v_x is the derivative of v wrt x
        @param v tensor of shape [batch_size, num_neural_networks, n_output_vel] if SVGD, or
                                 [batch_size, 1, n_output_vel] if HMC, represent our prediction of v
                                 (in isotropic case like here n_output_vel = 1 and v represent simply the conduction velocity)
        """

        # select the last dimension = 0 since we are in isotropic case and n_output_vel = 1
        v = v[:,:,0]    # for hmc has the shape 100,1,1; for svgd 100,30,1

        # store the sum of the square of each derivatives
        at_gradients_square = tf.zeros_like(at_gr[0])
        v_gradients_square = tf.zeros_like(v_gr[0])

        # add the sum of the square of each derivatives wrt to input (x, y(if n_input=2), z(if n_input=3))
        for at_gr_input in at_gr:
            at_gradients_square += tf.math.square(at_gr_input)

        for v_gr_input in v_gr:
            v_gradients_square += tf.math.square(v_gr_input)

        # compute the loss 1: residual of the PDE
        # ||Grad at|| = 1/v
        loss_1 = tf.math.multiply( tf.math.sqrt(at_gradients_square) , v) -1.

        # compute the loss 2: penalizing high gradients of Grad v
        # ||Grad v|| small enough
        loss_2 = self.param2loss*tf.math.sqrt(v_gradients_square)

        return loss_1, loss_2



class anisotropic_eikonal(pde_constraint):
    """
    Child class of pde_constraint

    Implement the anisotropic eikonal (2D)
    loss1 => anisotropic eikonal sqrt( Grad(at)^t * M * Grad(at) ) = 1

    where M is the conduction anisotropic tensor of the form:
    M = [a , -c]
        [-c,  b]

    loss2 => constraints between a,b and c
            { a*an_constraints[0] = b
            { a*an_constraints[1] = c
    """
    # Anisotropic case
    def __init__(self, par, anisotropic_constraints):
        """! Constructor
        @param par an object of our param class
        """

        """Use the parent constructor (pde_constraint.__init__)"""
        super().__init__(par.method, par.n_input, par.n_output_vel)

        # store the anisotropic constraints
        self.an_constraints = anisotropic_constraints

    def compute_pde_losses(self, at_gr, v_gr, v):
        """compute the pde losses (override parent method)
        @param at_gr list of tensors (at_x, at_y, at_z) (if 3D), where at_x is the derivative of at wrt x
        @param v_gr list of tensors (v_x, v_y, v_z) (if 3D), where v_x is the derivative of v wrt x
        @param v tensor of shape [batch_size, num_neural_networks, n_output_vel] if SVGD, or
                                 [batch_size, 1, n_output_vel] if HMC, represent our prediction of v
                                 (in anisotropic case like here n_output_vel > 1, and each slice in the third
                                 dimension represent a different Tensor entries a,b and c)
        """

        if(self.n_input == 2):
            # select each entries (a,b if n_output_vel = 2; a,b,c if n_output_vel = 3)
            a = v[:,:,0]
            b = v[:,:,1]

            # compute the loss 1: residual of the PDE
            loss_1 = tf.math.multiply(tf.math.square(at_gr[0]),a) + \
                     tf.math.multiply(tf.math.square(at_gr[1]),b)

            # compute the loss 2: imposing the anisotropic_constraints
            loss_2 = tf.math.abs(self.an_constraints[0]*a-b)

            # if n_output_vel = 3 add aslo the last entries of the tensors (c) and the relative anisotropic_constraint
            if(self.n_output_vel == 3):
                c = v[:,:,2]
                loss_1 -= 2*tf.math.multiply(tf.math.multiply(at_gr[0],at_gr[1]),c)
                loss_2 += tf.math.abs(self.an_constraints[1]*a-c)

            loss_1 -= 1.

            return loss_1, loss_2
        else:
            raise Exception("Only case 2D anisotropic implemented")

class laplace(pde_constraint):
    # Anisotropic case
    def __init__(self, par):

        super().__init__(par.method, par.n_input, par.n_output_vel)

    def compute_pde_losses(self, at_gr_2, v):
        """
        - Nabla(at) = v -> v+Nabla(at)=0
        """
        if(self.n_input == 1):
            # select each entries (a,b if n_output_vel = 2; a,b,c if n_output_vel = 3)
            v = v[:,:,0]

            # compute the loss 1: residual of the PDE
            loss_1 = at_gr_2[0] + v

            return loss_1
        else:
            raise Exception("Only case 1D laplace implemented")
