import numpy as np
import math
from collections.abc import Callable


class Data:
    def __init__(self, x_b, y_b_target=None):
        """
        :param x_b: Input data in batch ndarray(N(neurons)*N(batches))
        :param y_b_target: Target output data in batch 
        """
        self.x_b, self.x_b_size = x_b, x_b.shape[0]
        self.y_b_target = y_b_target


class Activation:
    
    @staticmethod
    def relu(z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    @staticmethod
    def softmax(z):
        """
        Softmax activation function
        Numerically stable implementation
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_derivative(z):
        """
        For use with categorical cross-entropy, we typically don't need this
        as the combined derivative simplifies
        """
        s = Activation.softmax(z)
        return s * (1 - s)


class Loss:
    
    @staticmethod
    def categorical_crossentropy(y_b_p, y_b_target):
        """
        Categorical cross-entropy loss
        :param y_b_p: Predicted probabilities (after softmax)
        :param y_b_target: One-hot encoded target labels
        :return: Loss per sample
        """
        # Clip predictions to prevent log(0)
        y_b_p_clipped = np.clip(y_b_p, 1e-15, 1 - 1e-15)
        # Calculate loss: -sum(y_target * log(y_pred))
        return -np.sum(y_b_target * np.log(y_b_p_clipped), axis=1, keepdims=True)
    
    @staticmethod
    def categorical_crossentropy_derivative(y_b_p, y_b_target):
        """
        Derivative of categorical cross-entropy with respect to predictions
        When combined with softmax, this simplifies to: y_pred - y_target
        """
        return y_b_p - y_b_target


class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, 
                 activation: Callable = Activation.relu,
                 da_dy: Callable = Activation.relu_derivative,
                 loss: Callable = Loss.categorical_crossentropy, 
                 dL_dy: Callable = Loss.categorical_crossentropy_derivative):
        
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        
        #initialization for ReLU layers
        self.weights_t = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros(n_neurons)
        
        self.x_b = None
        self.z_b = None
        self.weight_gradients_t = np.zeros((n_inputs, n_neurons))
        self.bias_gradient = np.zeros(n_neurons)
        self.y_b_p = None
        self.activation = activation
        self.da_dy = da_dy
        self.loss = loss
        self.dL_dy = dL_dy

    def calculate_output(self, x_b):
        self.x_b = x_b
        self.z_b = (self.x_b @ self.weights_t) + self.biases
        self.y_b_p = self.activation(self.z_b)
        return self.y_b_p

    def calculate_output_layer_neuron_values(self, y_b_target):
        """
        For softmax + categorical cross-entropy the gradient simplifies to:
        dL/dz = y_pred - y_target
        """
        # When using softmax with categorical cross-entropy the combined derivative is the difference
        neuron_values = self.dL_dy(self.y_b_p, y_b_target)
        return neuron_values
    
    def calculate_hidden_layer_neuron_values(self, old_neuron_values, old_layer):
        """
        Calculate neuron values for hidden layers
        dL_dz_current = dL_dz_old @ old_weights * dy/dz_current
        """
        dL_dy = old_neuron_values @ old_layer.weights_t.T
        dy_dz = self.da_dy(self.z_b)
        neuron_values = dL_dy * dy_dz
        return neuron_values
    
    def update_gradients(self, neuron_values):
        batch_size = self.x_b.shape[0]
        self.bias_gradient = neuron_values.sum(axis=0) / batch_size
        self.weight_gradients_t = (self.x_b.T @ neuron_values) / batch_size

    def apply_gradients(self, learning_rate: float):
        self.weights_t -= learning_rate * self.weight_gradients_t
        self.biases -= learning_rate * self.bias_gradient
    
    def clear_gradients(self):
        self.weight_gradients_t = np.zeros((self.n_inputs, self.n_neurons))
        self.bias_gradient = np.zeros(self.n_neurons)


class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Creates a neural network with:
        - ReLU activation in hidden layers
        - Softmax activation in output layer
        - Categorical cross-entropy loss
        
        :param layer_sizes: Size of all layers stored as a 1d array
        """
        self.layer_sizes = layer_sizes
        self.layers = []
        self.n_layers = len(layer_sizes)
        
        # Create hidden layers with ReLU
        for i in range(self.n_layers - 2):
            self.layers.append(
                Layer(
                    n_inputs=layer_sizes[i],
                    n_neurons=layer_sizes[i + 1],
                    activation=Activation.relu,
                    da_dy=Activation.relu_derivative,
                    loss=Loss.categorical_crossentropy,
                    dL_dy=Loss.categorical_crossentropy_derivative
                )
            )
        
        # Create output layer with Softmax
        if self.n_layers > 1:
            self.layers.append(
                Layer(
                    n_inputs=layer_sizes[-2],
                    n_neurons=layer_sizes[-1],
                    activation=Activation.softmax,
                    da_dy=Activation.softmax_derivative,
                    loss=Loss.categorical_crossentropy,
                    dL_dy=Loss.categorical_crossentropy_derivative
                )
            )

    def calculate_output(self, x_b):
        current_input_b = x_b
        for layer in self.layers:
            current_output_b = layer.calculate_output(current_input_b)
            current_input_b = current_output_b
        return current_output_b
    
    def avg_loss(self, data: Data):
        x_b = data.x_b
        y_b_target = data.y_b_target
        y_b_p = self.calculate_output(x_b)
        avg_loss = self.layers[-1].loss(y_b_p, y_b_target).mean()
        return avg_loss
    
    def update_gradients(self, y_b_target):
        # Output layer
        output_layer = self.layers[-1]
        neuron_values = output_layer.calculate_output_layer_neuron_values(y_b_target)
        output_layer.update_gradients(neuron_values)

        # Hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]
            old_layer = self.layers[i + 1]
            
            neuron_values = current_layer.calculate_hidden_layer_neuron_values(
                neuron_values,
                old_layer
            )
            current_layer.update_gradients(neuron_values)

    def apply_gradients(self, learning_rate):
        for layer in self.layers:
            layer.apply_gradients(learning_rate)
            layer.clear_gradients()

    def learn(self, data: Data, learning_rate: float):
        """Performs one full training step"""
        # Forward pass
        self.calculate_output(data.x_b)
        
        # Backward pass
        self.update_gradients(data.y_b_target)
        
        # Update parameters
        self.apply_gradients(learning_rate)
    
    def predict(self, x_b):
        """Get predicted class labels"""
        probabilities = self.calculate_output(x_b)
        return np.argmax(probabilities, axis=1)
    
    def accuracy(self, data: Data):
        """Calculate classification accuracy"""
        predictions = self.predict(data.x_b)
        targets = np.argmax(data.y_b_target, axis=1)
        return np.mean(predictions == targets)