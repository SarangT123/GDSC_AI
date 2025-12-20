import numpy as np
from math import exp


class DataPoint:
    pass


class Layer:
    def __init__(self,n_inputs,n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros(1,n_neurons)
        self.inputs = None
        self.weight_gradients = np.zeros((n_inputs, n_neurons))
        self.bias_gradients = np.zeros(n_neurons)

    def ReLU(self,output):
        relu = []
        for i in output:
            relu.append(0 if i<=0 else i)
        return relu

    def sigmoid(self,output):
        sigmoid = []
        for y in output:
            sigmoid.append(1/(1+exp(-y)))
        return sigmoid
    
    def sigmoid_activation_derivative(self,output):
        # y = 1/(1+e^(-x))
        # dy/dx = y(1-y)
        y = self.sigmoid(output)
        return y*(1-y)

    
    def square_neuron_cost(self,pred,target):
        error = pred-target
        return error **2 

    def square_neuron_cost_derivative(self,pred,target):
        # y = error
        # y=self.square_neuron_cost(pred,target)
        # instead of above just return 2*error which is 2(pred-target)
        return 2*(pred-target)



    def calculate_output(self,inputs):
        self.inputs = inputs
        self.output = self.sigmoid(np.dot(inputs,self.weights)+self.biases)
        return self.output



    def calculate_output_layer_neuron_values(self,expected_outputs):
        neuron_values = []
        length = len(expected_outputs)
        for i in range(0,length):

            # Eval PD for current node cost/activation and cost/output(z=xi*wi + bi)
            loss_derivative = self.square_neuron_cost_derivative(self.sigmoid[i],expected_outputs[i])
            activation_derivative_value = self.sigmoid_activation_derivative(self.sigmoid[i],expected_outputs[i])

        # neuron_values  =  da/dw * dc/da
        neuron_values.append(loss_derivative*activation_derivative_value)
        return neuron_values
    
    def calculate_hidden_layer_neuron_values(self,old_layer,old_neuron_values):
        new_neuron_values = np.zeros() #aka nnv for short
        nnv_length = self.n_neurons # num of neurons out will be length of nnv
        onv_length = len(old_neuron_values)
        for new_neuron_index in range(0,nnv_length):
            new_neuron_value = 0
            for old_neuron_index in range(0,onv_length):
                # partial derivative of output(z) wrt to the input a
                dz_da = old_layer.weights[new_neuron_index][old_neuron_index]
                new_neuron_value +=dz_da*old_neuron_values[old_neuron_index]

            new_neuron_value = new_neuron_value*self.sigmoid_activation_derivative(self.output[new_neuron_index])
            new_neuron_values[new_neuron_index]=new_neuron_value

        return new_neuron_values


    
    def update_gradients(self,neuron_values):
        for neuron_out in range(0,self.n_neurons):
            for neuron_in in range(0,self.n_inputs):
                dz_dw = self.inputs[neuron_in]
                # Evaluating the Derivative of cost/loss wrt weight
                dL_dw = dz_dw * neuron_values # neuron_values  =  da/dw * dc/da


                # changing the gradient accordingly 
                self.weight_gradients[neuron_in][neuron_out] += dL_dw

            # Evaluate partial derivative of dL/db of the current node
            # dL/db = dz/db x da/db x dl/db (remember a is input or the activated output)
            # z = aw + b imples dz/db = 1
            # dL/dw = 1* neuron_values 
            dL_db =  neuron_values
            self.bias_gradients[neuron_out] += dL_db

    def apply_gradients(self,learn_rate):
        for neuron_out in range(0,self.n_neurons):
            self.biases[neuron_out] -= self.bias_gradients[neuron_out]*learn_rate
            for neuron_in in range(0,self.n_inputs):
                self.weights[neuron_out][neuron_in] -= self.weight_gradients[neuron_in][neuron_out] * learn_rate
    def clear_gradients(self):
        # Reset all gradients to zero
        self.weight_gradients = np.zeros((self.n_inputs, self.n_neurons))
        self.bias_gradients = np.zeros(self.n_neurons)




class NeuralNetwork:
    def __init__(self,layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = []
        self.n_layers = len(layer_sizes)
        for i in range(0,self.n_layers-1):
            self.layers.append(Layer(layer_sizes[i],layer_sizes[i+1]))
    
    def calculate_output(self,input=None):
        input = input if input is not None else self.layers
        self.outputs = []
        for layer in input:
            self.outputs.append(layer.calculate_output())
        return self.outputs
    

    def loss_single(self,data_point):
        # data_point is a single set of input thats it is a list of inputs the same length as the input neurons
        outputs = self.calculate_output(data_point.inputs)
        output_layer  = self.layers[self.n_layers-1]
        loss = 0
        for neuron in range(0,len(outputs)):
            loss +=output_layer.square_neuron_cost(outputs[neuron],data_point.expected_outputs[neuron])
        return loss
    
    def loss(self,data):
        total_cost = 0
        for data_point in data:
            total_cost += self.loss_single(data_point)

    # Reminder : document this functions working in an external file later 
    def update_all_gradients(self,data_point):
        # Run input through the network so that each layer will store values 
        self.calculate_output(data_point.inputs)

        # update gradients of output layer
        output_layer = self.layers[self.n_layers-1]
        neuron_values = output_layer.calculate_output_layer_neuron_values(data_point.expected_outputs)
        output_layer.update_gradients(neuron_values)

        # loop through all hidden layers and update gradients 
        for hidden_layer_index in range(self.n_layers-2, 0,-1):
            #update gradients of the hidden layer
            hidden_layer = self.layers[hidden_layer_index]
            neuron_values = hidden_layer.calculate_hidden_layer_neuron_values(self.layers[hidden_layer_index+1],neuron_values)
            hidden_layer.update_gradients(neuron_values) 


    

    def learn(self,training_batch,learn_rate):


        for data_point in training_batch:
            self.update_all_gradients(data_point)

        for layer in self.layers:
            layer.apply_gradients(learn_rate / len(training_batch))
            layer.clear_gradients()
    
    

    
        

    def classify():
        pass


    


