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


class SoftmaxCrossEntropy:
    EPSILON = 1e-9

    def forward(self, logits, y_true_one):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exps / np.sum(exps, axis=1, keepdims=True)
        self.probs = probabilities
        self.y_true = y_true_one

        m = y_true_one.shape[0]
        loss = -np.sum(y_true_one * np.log(probabilities + self.EPSILON)) / m
        return loss

    def backward(self):
        m = self.y_true.shape[0]
        return (self.probs - self.y_true) / m


class MLP:
    def __init__(self):
        self.layers = []

    def add(self, layer: DenseLayer):
        self.layers.append(layer)

    def forward(self, data):
        curr = data
        for layer in self.layers:
            curr = layer.forward(curr)
        return curr

    def train(self, training_data, y_data, labels_val=None, y_val=None, epochs=1000, batch_size=32):
        loss_fn = SoftmaxCrossEntropy()
        n_samples = training_data.shape[0]

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            training_data_shuffled = training_data[indices]
            y_data_shuffled = y_data[indices]

            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                data_batch = training_data_shuffled[i : i + batch_size]
                y_batch = y_data_shuffled[i : i + batch_size]

                output = self.forward(data_batch)

                loss = loss_fn.forward(output, y_batch)
                grad = loss_fn.backward()

                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                epoch_loss += loss * data_batch.shape[0]

            if epoch % 50 == 0:
                avg_loss = epoch_loss / n_samples
                log_msg = f"Epoch {epoch}, Average Train Loss: {avg_loss:.4f}"

                if labels_val is not None and y_val is not None:
                    val_loss, val_accuracy = self.evaluate(labels_val, y_val)
                    log_msg += f" | Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%"

                print(log_msg)

    def evaluate(self, labels_data, y_data):
        loss_fn = SoftmaxCrossEntropy()

        output = self.forward(labels_data)

        loss = loss_fn.forward(output, y_data)

        predictions = np.argmax(output, axis=1)
        true_labels = np.argmax(y_data, axis=1)

        accuracy = np.mean(predictions == true_labels)

        return loss, accuracy
