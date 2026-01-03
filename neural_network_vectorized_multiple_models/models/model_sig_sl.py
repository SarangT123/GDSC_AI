"""
Model that uses sigmoid and square loss
"""

import numpy as np
import math
from collections.abc import Callable


#TODO: Explain new stuff in vectorization in readme


# Naming convention:
# Class names: PascalCase
# Function and variable names: snake_case
# x = input
# y = output
# _i = i'th thing
# y_pred = prediction/output from nn or _p
# y_target = actual values 
# _b = batch
# N(x) = number of x's
# n_ = Number of x's
#_t = transpose
# a = activation(y_b_p)


class Data:
    def __init__(self,x_b,y_b_target=None):
        """
        Docstring for __init__
        
        :param x_b: Input data in batch ndarray(N(neurons)*N(batches))
        :param y_b_target: Target output data in batch 
        """

        self.x_b,self.x_b_size = x_b,x_b.shape[0]
        self.y_b_target = y_b_target

class Activation:
    @staticmethod
    def sigmoid(z):
        """
        Docstring for sigmoid
        
        :param y_b_p: y_b_p is a matrix of order N(batches) x N(neurons)
        :return: matrix of order N(batches) x N(neurons)
        """
        return  1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(z):
        """
        Derivative of sigmoid activation
        
        :param z: PRE-ACTIVATION values
        :return: Derivative values
        """
        s = Activation.sigmoid(z)
        return s*(1-s)


class Loss:
    @staticmethod
    def square_loss(y_b_p,y_b_target):
        """
        Docstring for square_loss
        
        :param y_b_p: Matrix of order N(batches) x N(neurons) with outputs from output layer for x_b of same batch size
        :param y_b_target: Matrix of order N(batches) x N(neurons) with target values for x_b of the same batch size
        :return: Matrix of order N(batches) x N(neurons) with the calculated loss 
        """
        error = y_b_p-y_b_target
        return error**2
    

    @staticmethod
    def square_loss_derivative(y_b_p,y_b_target):
        return 2*(y_b_p-y_b_target)
    

class Layer:
    def __init__(self,n_inputs:int,n_neurons:int,activation:Callable = Activation.sigmoid,da_dy:Callable=Activation.sigmoid_derivative,loss:Callable = Loss.square_loss, dL_dy:Callable=Loss.square_loss_derivative):

        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights_t = np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros(n_neurons) # we will use broadcasting
        self.x_b = None
        self.weight_gradients_t = np.zeros((n_inputs,n_neurons))
        self.bias_gradient = np.zeros(n_neurons)
        self.y_b_p = None
        self.activation  = activation
        self.da_dy = da_dy
        self.loss = loss
        self.dL_dy = dL_dy

    def calculate_output(self, x_b):
        self.x_b=x_b
        self.z_b = (self.x_b@self.weights_t)+self.biases
        self.y_b_p = self.activation(self.z_b)
        return self.y_b_p

    def calculate_output_layer_neuron_values(self,y_b_target):
        """
        Docstring for calculate_output_layer_neuron_values
        calculates the neuron values for output layer
        that is dL/dz for each neuron in the output layer
        
        :param self: Description
        :param y_b_target: Matrix of order N(batches) x N(neurons) with target values for x_b of the same batch size
        """
        dL_dy = self.dL_dy(self.y_b_p, y_b_target)
        dy_dz = self.da_dy(self.z_b)
        neuron_values = dL_dy * dy_dz
        return neuron_values
    
    def calculate_hidden_layer_neuron_values(self,old_neuron_values,old_layer):
        """
        Docstring for calculate_hidden_layer_neuron_values
        dL_dz_current = dL_dz_old @ old_weights * dy/dz_current
        
        :param self: Description
        :param old_neuron_values: Description
        :param old_layer: Description
        """
        # TODO: explain this formula in readme
        dL_dy = old_neuron_values @ old_layer.weights_t.T
        dy_dz = self.da_dy(self.z_b)
        neuron_values = dL_dy * dy_dz
        return neuron_values
    
    def update_gradients(self,neuron_values):
        """
        Docstring for update_gradients
        
        :param self: Description
        :param neuron_values: Description
        """
        batch_size = self.x_b.shape[0]
        self.bias_gradient = neuron_values.sum(axis=0)/batch_size
        self.weight_gradients_t = (self.x_b.T @ neuron_values)/batch_size

    def apply_gradients(self,learning_rate:float):
        """
        Docstring for apply_gradients
        
        :param self: Description
        :param learning_rate: Description
        """
        self.weights_t -= learning_rate * self.weight_gradients_t
        self.biases -= learning_rate * self.bias_gradient
    def clear_gradients(self):
        """
        Docstring for clear_gradients
        Resets gradients to zero
        :param self: Description
        """
        self.weight_gradients_t = np.zeros((self.n_inputs,self.n_neurons))
        self.bias_gradient = np.zeros(self.n_neurons)

    



        

        




class NeuralNetwork:
    def __init__(self,layer_sizes,activation:Callable = Activation.sigmoid,da_dy:Callable=Activation.sigmoid_derivative,loss:Callable = Loss.square_loss, dL_dy:Callable=Loss.square_loss_derivative):
        """
        Docstring for __init__
        
        :param self: Description
        :param layer_sizes: Size of all layers stored as a 1d array
        """
        self.layer_sizes = layer_sizes
        self.layers = []
        self.n_layers = len(layer_sizes)
        for i in range(0,self.n_layers - 1):
            self.layers.append(Layer(n_inputs=layer_sizes[i],n_neurons=layer_sizes[i+1],activation=activation,da_dy=da_dy,loss=loss,dL_dy=dL_dy))

    def calculate_output(self,x_b):
        current_input_b = x_b
        for layer in self.layers:
            current_output_b = layer.calculate_output(current_input_b)
            current_input_b = current_output_b
        return current_output_b
    
    def avg_loss(self,data:Data):
        x_b = data.x_b
        y_b_target = data.y_b_target
        y_b_p = self.calculate_output(x_b)
        batch_size = x_b.shape[0]
        avg_loss = self.layers[-1].loss(y_b_p,y_b_target).mean()
        return avg_loss
    
    def update_gradients(self,y_b_target):

        #Output layer
        output_layer = self.layers[-1]
        neuron_values = output_layer.calculate_output_layer_neuron_values(y_b_target)
        output_layer.update_gradients(neuron_values)

        #Hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]
            old_layer = self.layers[i + 1]

            neuron_values = current_layer.calculate_hidden_layer_neuron_values(
                neuron_values,
                old_layer
            )
            current_layer.update_gradients(neuron_values)

    def apply_gradients(self,learning_rate):
        for layer in self.layers:
            layer.apply_gradients(learning_rate)
            layer.clear_gradients()

    def learn(self, data: Data, learning_rate: float):
        """
        Performs one full training step
        """
        # Forward pass
        self.calculate_output(data.x_b)

        # Backward pass
        self.update_gradients(data.y_b_target)

        # Update parameters
        self.apply_gradients(learning_rate)




        





    
