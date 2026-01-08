import numpy as np
from activation_functions import ActivationFunctions


class Layer:
    def __init__(self):
        self.trainable = False


class DenseLayer(Layer):
    """
        @param input_count - inputs / number of neurons in previous layer
        @param output_count - outputs / number of neurons in currnet layer
        @param activation_func - certain activation function for the layer
        @param learning rate - additional parameter for optimisation
    """
    def __init__(self, input_count, output_count, activation_func: ActivationFunctions, learning_rate=0.01):
        super().__init__()
        self.trainable = True
        self.weights = np.random.randn(input_count, output_count) * np.sqrt(2. / input_count)
        self.biases = np.zeros((1, output_count))
        self.activation_func = activation_func

    def forward(self, input_data):
        self.input = input_data

        # Calculating linear part of each neuron
        self.linear_part = np.dot(input_data, self.weights) + self.biases

        # Processing outputs with activation function
        self.output = self.activation_func.forward(self.linear_part)

        return self.output

    def backward(self, grad_output):
        derivative_activation = grad_output * self.activation_func.backward(self.linear_part)

        self.grad_weights = np.dot(self.input.T, derivative_activation)
        self.grad_biases = np.sum(derivative_activation, axis=0, keepdims=True)

        grad_input = np.dot(derivative_activation, self.weights.T)

        return grad_input


class LayerDropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
        return self.dinputs
