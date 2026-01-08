from abc import ABC, abstractmethod
import numpy as np


class Optimiser(ABC):
    def __init__(self, learning_rate=1., decay=0., l1=0., l2=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.l1 = l1
        self.l2 = l2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def post_update_params(self):
        self.iterations += 1

    def apply_regularization(self, layer):
        if self.l1 > 0:
            layer.grad_weights += self.l1 * np.sign(layer.weights)
            layer.grad_biases += self.l1 * np.sign(layer.biases)

        if self.l2 > 0:
            layer.grad_weights += self.l2 * 2 * layer.weights
            layer.grad_biases += self.l2 * 2 * layer.biases

    @abstractmethod
    def step(self, layer):
        pass


class SGD(Optimiser):
    def __init__(self, learning_rate=1.0, decay=0., momentum=0., l1=0., l2=0.):
        super().__init__(learning_rate, decay, l1, l2)
        self.momentum = momentum

    def step(self, layer):
        self.apply_regularization(layer)

        weight_updates = -self.current_learning_rate * layer.grad_weights
        bias_updates = -self.current_learning_rate * layer.grad_biases

        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates += self.momentum * layer.weight_momentums
            bias_updates += self.momentum * layer.bias_momentums

            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates

        layer.weights += weight_updates
        layer.biases += bias_updates


class Adam(Optimiser):
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999, l1=0., l2=0.):
        super().__init__(learning_rate, decay, l1, l2)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.state = {}

    def step(self, layer):
        self.apply_regularization(layer)

        if layer not in self.state:
            self.state[layer] = { 't': 0, 
                                  'm_w': np.zeros_like(layer.weights), 'v_w': np.zeros_like(layer.weights),
                                  'm_b': np.zeros_like(layer.biases),  'v_b': np.zeros_like(layer.biases) }

        state = self.state[layer]
        state['t'] += 1
        t = state['t']

        layer.weights, state['m_w'], state['v_w'] = self._apply_adam(
            layer.weights, layer.grad_weights, state['m_w'], state['v_w'], t
        )
        layer.biases, state['m_b'], state['v_b'] = self._apply_adam(
            layer.biases, layer.grad_biases, state['m_b'], state['v_b'], t
        )

    def _apply_adam(self, params, grads, m, v, t):
        m = self.beta1 * m + (1 - self.beta1) * grads
        v = self.beta2 * v + (1 - self.beta2) * (grads ** 2)
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)
        return params - self.current_learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon), m, v