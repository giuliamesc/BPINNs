from .Equation  import Equation
from .Operators import Operators

class Laplace(Equation):
    """
    Laplace pde constraint implementation
    """
    def __init__(self, par):
        super().__init__(par)
        self.mu = Operators.tf_convert(par.physics["diffusion"])

    def comp_residual(self, inputs, out_sol, out_par, tape):
        u_list = Operators.tf_unpack(out_sol)
        lap_u  = Operators.laplacian_vector(tape, u_list, inputs)
        lap_u  = Operators.tf_pack(lap_u)
        return lap_u * self.mu + out_par

    def comp_process(self, dataset):
        params = dict()
        return params

    def _normalize_data(self, vec):
        u_span = max(vec[1])-min(vec[1])
        f_span = max(vec[2])-min(vec[2])
        new_dom_data = (vec[0], vec[1]/u_span, vec[2]/f_span)
        return new_dom_data

    def data_process(self, dataset, params):
        """ Normalization of u and f """
        new_dataset = dataset
        """
        new_dataset.dom_data   = self._normalize_data(dataset.dom_data)
        new_dataset.exact_data = self._normalize_data(dataset.exact_data)
        new_dataset.coll_data  = self._normalize_data(dataset.coll_data)
        new_dataset.noise_data = self._normalize_data(dataset.noise_data)
        """
        return new_dataset

    def pre_process(self, inputs, params):
        """ Pre-process in Laplace problem is the identity transformation """
        return inputs

    def post_process(self, outputs, params):
        """ Post-process in Laplace problem is the identity transformation """
        """
        span_u = max(outputs[0])-min(outputs[0])
        span_f = max(outputs[1])-min(outputs[1])
        new_output_u = outputs[0] * 2
        new_output_f = outputs[1] * 120
        new_outputs = (new_output_u, new_output_f)
        """
        return outputs