from activation_functions import ActivationFunctions, ReLU, Sigmoid
import numpy as np


class DenseLayer:
    """
        @param input_count - inputs / number of neurons in previous layer
        @param output_count - outputs / number of neurons in currnet layer
        @param activation_func - certain activation function for the layer
        @param learning rate - additional parameter for optimisation
    """
    def __init__(self, input_count, output_count, activation_func: ActivationFunctions, learning_rate=0.01):
        self.learining_rate = learning_rate
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

        grad_weights = np.dot(self.input.T, derivative_activation)
        grad_biases = np.sum(derivative_activation, axis=0, keepdims=True)

        grad_input = np.dot(derivative_activation, self.weights.T)

        self.weights -= self.learining_rate * grad_weights
        self.biases -= self.learining_rate * grad_biases

        return grad_input


class MLP:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, data):
        curr = data
        for layer in self.layers:
            curr = layer.forward(curr)
        return curr

    def train(self, batch_to_process, y_one_hot, epochs=1000):
        pass