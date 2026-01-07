from abc import ABC, abstractmethod
from layer import DenseLayer
import numpy as np


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


class Adam(Optimiser):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.state = {}

    def step(self, layer: DenseLayer):
        if layer not in self.state:
            # moments and velocities for weights and biases
            self.state[layer] = {
                'm_w': np.zeros_like(layer.weights),
                'v_w': np.zeros_like(layer.weights),
                'm_b': np.zeros_like(layer.biases),
                'v_b': np.zeros_like(layer.biases)
            }

        state = self.state[layer]

        # incrementation of timer; should be incremented after one epoch, but it can be as it is
        self.t += 1
        t = self.t

        # calculating momentum and velocity for updateing Momentum and RMSprop for weights
        state['m_w'] = self.beta1 * state['m_w'] + (1 - self.beta1) * layer.grad_weights
        state['v_w'] = self.beta2 * state['v_w'] + (1 - self.beta2) * (layer.grad_weights ** 2)

        # bias correction - "cold start" prevention
        m_w_hat = state['m_w'] / (1 - self.beta1 ** t)
        v_w_hat = state['v_w'] / (1 - self.beta2 ** t)

        # updating weights in given layer
        layer.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        # similar operations as above, but now for biases
        state['m_b'] = self.beta1 * state['m_b'] + (1 - self.beta1) * layer.grad_biases
        state['v_b'] = self.beta2 * state['v_b'] + (1 - self.beta2) * (layer.grad_biases ** 2)

        m_b_hat = state['m_b'] / (1 - self.beta1 ** t)
        v_b_hat = state['v_b'] / (1 - self.beta2 ** t)

        # updating biases in given layer
        layer.biases -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
