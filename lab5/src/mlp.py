from activation_functions import ActivationFunctions
from layer import Layer
from loss_functions import LossFunctions
from optimiser import Optimiser
import numpy as np


class MLP:
    def __init__(self):
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, data):
        curr = data
        for layer in self.layers:
            curr = layer.forward(curr)
        return curr

    def train(self,
              training_data,
              y_data,
              loss_fn: LossFunctions,
              optimiser: Optimiser,
              labels_val=None,
              y_val=None,
              epochs=1000,
              batch_size=32):

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
                for layer in self.layers:
                    if layer.trainable:
                        optimiser.step(layer)

                epoch_loss += loss * data_batch.shape[0]

            if epoch % 50 == 0:
                avg_loss = epoch_loss / n_samples
                log_msg = f"Epoch {epoch}, Average Train Loss: {avg_loss:.4f}"

                if labels_val is not None and y_val is not None:
                    val_loss, val_accuracy = self.evaluate(labels_val, y_val, loss_fn)
                    log_msg += f" | Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%"

                print(log_msg)

    def evaluate(self, labels_data, y_data, loss_fn: LossFunctions):
        output = self.forward(labels_data)

        loss = loss_fn.forward(output, y_data)

        predictions = np.argmax(output, axis=1)
        true_labels = np.argmax(y_data, axis=1)

        accuracy = np.mean(predictions == true_labels)

        return loss, accuracy
