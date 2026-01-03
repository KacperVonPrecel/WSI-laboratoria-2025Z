import numpy as np
from abc import ABC, abstractmethod


class ActivationFunctions(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass


class ReLU(ActivationFunctions):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return x > 0


class Sigmoid(ActivationFunctions):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)


class Linear(ActivationFunctions):
    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)
