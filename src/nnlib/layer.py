import numpy as np
from collections.abc import Callable

from nnlib.activations import relu, relu_derivative


def he_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)


def xavier_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * np.sqrt(1.0 / fan_in)


def normal_init(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out)


INITIALIZERS = {
    'he': he_init,
    'xavier': xavier_init,
    'normal': normal_init,
}


class Layer:
    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        activation: Callable = relu,
        da_dz: Callable = relu_derivative,
        dL_da: Callable | None = None,
        dL_dz_output: Callable | None = None,
        weight_init: str = 'he',
    ):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        init_fn = INITIALIZERS.get(weight_init, normal_init)
        self.weights = init_fn(n_inputs, n_neurons)
        self.biases = np.zeros(n_neurons)

        self.x_b = None
        self.z_b = None
        self.y_b_p = None

        self.weight_gradients = np.zeros((n_inputs, n_neurons))
        self.bias_gradient = np.zeros(n_neurons)

        self.activation = activation
        self.da_dz = da_dz
        self.dL_da = dL_da
        self.dL_dz_output = dL_dz_output

    def forward(self, x_b):
        self.x_b = x_b
        self.z_b = (self.x_b @ self.weights) + self.biases
        self.y_b_p = self.activation(self.z_b)
        return self.y_b_p

    def output_neuron_values(self, y_b_target):
        if self.dL_dz_output is not None:
            return self.dL_dz_output(self.y_b_p, y_b_target)
        dL_da = self.dL_da(self.y_b_p, y_b_target)
        da_dz = self.da_dz(self.z_b)
        return dL_da * da_dz

    def hidden_neuron_values(self, neuron_values_next, layer_next):
        dL_da = neuron_values_next @ layer_next.weights.T
        da_dz = self.da_dz(self.z_b)
        return dL_da * da_dz

    def update_gradients(self, neuron_values):
        batch_size = self.x_b.shape[0]
        self.bias_gradient = neuron_values.sum(axis=0) / batch_size
        self.weight_gradients = (self.x_b.T @ neuron_values) / batch_size

    def apply_gradients(self, learning_rate: float):
        self.weights -= learning_rate * self.weight_gradients
        self.biases -= learning_rate * self.bias_gradient

    def clear_gradients(self):
        self.weight_gradients = np.zeros((self.n_inputs, self.n_neurons))
        self.bias_gradient = np.zeros(self.n_neurons)

    def get_parameters(self):
        return {'weights': self.weights.copy(), 'biases': self.biases.copy()}

    def set_parameters(self, weights, biases):
        self.weights = weights.copy()
        self.biases = biases.copy()
