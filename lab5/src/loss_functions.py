import numpy as np
from abc import ABC, abstractmethod


class LossFunctions(ABC):
    EPSILON = 1e-9

    @abstractmethod
    def forward(self, logits, y_true_one_hot):
        pass

    @abstractmethod
    def backward(self):
        pass


class SoftmaxCrossEntropy(LossFunctions):

    def forward(self, logits, y_true_one_hot):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exps / np.sum(exps, axis=1, keepdims=True)
        self.probs = probabilities
        self.y_true = y_true_one_hot

        m = y_true_one_hot.shape[0]
        loss = -np.sum(y_true_one_hot * np.log(probabilities + self.EPSILON)) / m
        return loss

    def backward(self):
        m = self.y_true.shape[0]
        return (self.probs - self.y_true) / m
