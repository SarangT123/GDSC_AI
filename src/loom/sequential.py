from collections.abc import Callable

import numpy as np

from loom.activations import relu, relu_derivative, softmax, softmax_derivative
from loom.data import Data
from loom.layer import Layer
from loom.losses import categorical_crossentropy, categorical_crossentropy_derivative


class Sequential:
    def __init__(
        self,
        layer_sizes: list[int],
        activation: Callable = relu,
        da_dz: Callable = relu_derivative,
        output_activation: Callable = softmax,
        output_da_dz: Callable = softmax_derivative,
        dL_da: Callable | None = None,
        dL_dz_output: Callable | None = None,
        loss_fn: Callable = categorical_crossentropy,
        weight_init: str = 'he',
    ):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.layers = []
        self.loss_fn = loss_fn
        self._dL_da = dL_da
        self._dL_dz_output = dL_dz_output

        if dL_da is None:
            self._dL_da = categorical_crossentropy_derivative
        if dL_dz_output is None:
            self._dL_dz_output = categorical_crossentropy_derivative

        for i in range(self.n_layers - 2):
            self.layers.append(
                Layer(
                    n_inputs=layer_sizes[i],
                    n_neurons=layer_sizes[i + 1],
                    activation=activation,
                    da_dz=da_dz,
                    dL_da=self._dL_da,
                    weight_init=weight_init,
                )
            )

        if self.n_layers > 1:
            self.layers.append(
                Layer(
                    n_inputs=layer_sizes[-2],
                    n_neurons=layer_sizes[-1],
                    activation=output_activation,
                    da_dz=output_da_dz,
                    dL_da=self._dL_da,
                    dL_dz_output=self._dL_dz_output,
                    weight_init=weight_init,
                )
            )

    def forward(self, x_b):
        current = x_b
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def avg_loss(self, data: Data):
        y_pred = self.forward(data.x_b)
        return self.loss_fn(y_pred, data.y_b_target).mean()

    def backward(self, y_b_target):
        output_layer = self.layers[-1]
        neuron_values = output_layer.output_neuron_values(y_b_target)
        output_layer.update_gradients(neuron_values)

        for i in range(len(self.layers) - 2, -1, -1):
            current = self.layers[i]
            next_layer = self.layers[i + 1]
            neuron_values = current.hidden_neuron_values(neuron_values, next_layer)
            current.update_gradients(neuron_values)

    def step(self, learning_rate: float):
        for layer in self.layers:
            layer.apply_gradients(learning_rate)
            layer.clear_gradients()

    def learn(self, data: Data, learning_rate: float):
        self.forward(data.x_b)
        self.backward(data.y_b_target)
        self.step(learning_rate)

    def predict(self, x_b):
        probabilities = self.forward(x_b)
        if probabilities.shape[1] == 1:
            return (probabilities > 0.5).astype(int).flatten()
        return np.argmax(probabilities, axis=1)

    def accuracy(self, data: Data):
        predictions = self.predict(data.x_b)
        if data.y_b_target.shape[1] == 1:
            targets = data.y_b_target.flatten()
        else:
            targets = np.argmax(data.y_b_target, axis=1)
        return np.mean(predictions == targets)

    def get_parameters(self):
        return [layer.get_parameters() for layer in self.layers]

    def set_parameters(self, params_list):
        for layer, params in zip(self.layers, params_list):
            layer.set_parameters(params['weights'], params['biases'])
