from abc import ABC, abstractmethod
from layer import DenseLayer


class Optimiser(ABC):
    @abstractmethod
    def step(self, layer):
        pass


class SGD(Optimiser):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layer: DenseLayer):
        layer.weights -= self.learning_rate * layer.grad_weights
        layer.biases -= self.learning_rate * layer.grad_biases
