from .Equation  import Equation
import tensorflow as tf

class Regression(Equation):
    """
    Regression implementation
    """
    def __init__(self, par):
        super().__init__(par)

    def solution(self, u_tilde, *_):
        return u_tilde

    def parametric_field(self, u_tilde, *_):
        return u_tilde

    def comp_process(self, dataset):
        params = dict()
        return params

    def data_process(self, dataset, params):
        """ TO BE DONE """
        new_dataset = dataset
        return new_dataset

    def pre_process(self, inputs, params):
        """ Pre-process in Laplace problem is the identity transformation """
        return inputs

    def post_process(self, outputs, params):
        """ Post-process in Laplace problem is the identity transformation """
        return outputs