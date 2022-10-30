from .PhysNN import PhysNN
import tensorflow as tf
import numpy as np

class PredNN(PhysNN):
    """
    - Computes a sample given thetas and inputs
    - Outputs a prediction given samples of theta
    - Post-processing utilities:
        - Error computation
        - Uncertainity Quantification
    """
    
    def __init__(self, **kw):
        
        super(PredNN, self).__init__(**kw)
        self.thetas = list() # Empty list where samples of network parameters will be stored

    def __compute_sample(self, theta, inputs):
        """ Computes output sampling with one given theta """
        self.nn_params = theta
        u, f = self.forward(inputs)
        u_denorm = u * self.u_coeff[1] + self.u_coeff[0]
        f_denorm = f * self.f_coeff[1] + self.f_coeff[0] 
        return u_denorm, f_denorm
    
    def __predict(self, inputs, n_thetas = None):
        """ 
        Computes a list of output sampling from the list of thetas
        out_sol: list with len n_thetas, of tf tensor (n_sample, n_out_sol)
        out_par: list with len n_thetas, of tf tensor (n_sample, n_out_par)
        """
        out_sol = list()
        out_par = list()

        if n_thetas is None: n_thetas = len(self.thetas)
        if n_thetas > len(self.thetas): raise("Not enough thetas!!!")
        for theta in self.thetas[-n_thetas:]:
            outputs = self.__compute_sample(theta, inputs)
            out_sol.append(outputs[0])
            out_par.append(outputs[1])

        return out_sol, out_par

    def __statistics(self, output):
        """
        Returns mean and standard deviation of a list of tensors
        mean : (n_samples, output[0].shape[1])
        std  : (n_samples, output[0].shape[1])
        """
        mean = np.mean(output, axis = 0)
        std = np.std(output, axis = 0)
        return mean, std

    def __compute_UQ(self, function_confidence):
        """ 
        Returns a dictionary containing quantifications of the uncertainity of results:
        - Mean standard deviation on solution and parametric field
        - Max standard deviation on solution and paramteric field
        """
        u_q = {
            "uq_sol_mean" : np.mean(function_confidence["sol_std"], axis = 0),
            "uq_par_mean" : np.mean(function_confidence["par_std"], axis = 0),
            "uq_sol_max"  : np.max(function_confidence["sol_std"], axis = 0),
            "uq_par_max"  : np.max(function_confidence["par_std"], axis = 0)}
        return u_q

    def __metric(self, x, y):
        """ Component-wise MSE """
        metric = tf.keras.losses.MSE 
        return [metric(x[:,i],y[:,i]).numpy() for i in range(x.shape[1])]

    def __compute_errors(self, function_confidence, dataset):
        """ Computes errors on the solution and parametric field """
        sol_true, par_true = dataset.dom_data[1], dataset.dom_data[2]
        sol_NN, par_NN = function_confidence["sol_NN"], function_confidence["par_NN"]

        error_sol = self.__metric(sol_NN, sol_true)
        error_par = self.__metric(par_NN, par_true)
        norm_sol  = self.__metric(sol_true, tf.zeros_like(sol_true))
        norm_par  = self.__metric(par_true, tf.zeros_like(par_true))

        err = {
            "error_sol" : np.divide(error_sol,norm_sol),
            "error_par" : np.divide(error_par,norm_par)
            }

        return err
    
    def mean_and_std(self, inputs):
        """
        Computes mean and standard deviation of the output samples
        functions_confidence: dictionary containing outputs of __statistics() for out_sol and out_par
        """
        out_sol, out_par = self.__predict(inputs)
        mean_sol, std_sol = self.__statistics(out_sol)
        mean_par, std_par = self.__statistics(out_par)
        functions_confidence = {"sol_NN": mean_sol, "sol_std": std_sol, 
                                "par_NN": mean_par, "par_std": std_par}
        return functions_confidence

    def draw_samples(self, inputs):
        """ Draws samples of the solution and of the parametric field given inputs """
        out_sol, out_par = self.__predict(inputs)
        out_sol = [value.numpy() for value in out_sol]
        out_par = [value.numpy() for value in out_par]
        functions_nn_samples = {"sol_samples": out_sol, "par_samples": out_par}
        return functions_nn_samples

    def test_errors(self, function_confidence, dataset):
        """ Creation of a dictionary containing errors and UQ """
        err = self.__compute_errors(function_confidence, dataset)
        u_q = self.__compute_UQ(function_confidence)
        return err | u_q

    def fill_thetas(self, new_thetas):
        """ Initializes the list of thetas with a given set """
        if not self.thetas.empty:
            raise Warning("Some thetas have been deleted!") 
        self.thetas = new_thetas

    def __disp_UQ(self, tag, means, maxs): # VALUTA SE CONSIDERARE IN RELAZIONE AL NOISE
        form = lambda value: f"{100*value:1.2f}%"
        labels = ["x", "y", "z"][:len(means)]
        for label, mean, mx in zip(labels, means, maxs):
            if    len(means) == 1:     print(f"\t{tag} UQ: (mean) {form(mean)} - (max) {form(mx)}")
            else: print(f"\t{tag} UQ in direction {label}: (mean) {form(mean)} - (max) {form(mx)}")

    def __disp_err(self, tag, errors):
        form = lambda value: f"{100*value:1.2f}%"
        labels = ["x", "y", "z"][:len(errors)]
        for label, error in zip(labels, errors):
            if    len(errors) == 1:    print(f"\t{tag} error: {form(error)}")
            else: print(f"\t{tag} error in direction {label}: {form(error)}")

    def show_errors(self, errors):
        """ Prints on terminal the errors computed above """
        self.__disp_err("Sol", errors["error_sol"])
        self.__disp_err("Par", errors["error_par"])
        self.__disp_UQ( "Sol", errors["uq_sol_mean"], errors["uq_sol_max"])
        self.__disp_UQ( "Par", errors["uq_par_mean"], errors["uq_par_max"])
