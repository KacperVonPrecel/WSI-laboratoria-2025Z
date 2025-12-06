import numpy as np


class GradientDescentAckley:
    def __init__(self, step_size=0.1, max_iter=1000, tol=1e-4):
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol

    def ackley_1d(self, x):
        return -20 * np.exp(-0.2*np.abs(x)) - np.exp(np.cos(2 * np.pi * x)) + 20 + np.exp(1)

    def gradient_ackley_1d(self, x):
        if np.abs(x) < 1e-10:
            sign = 0
        else:
            sign = x / np.abs(x)

        return 4 * sign * np.exp(-0.2 * np.abs(x)) + 2 * np.pi * np.sin(2 * np.pi * x) * np.exp(np.cos(2 * np.pi * x))

    def optimize_ackley_1d(self, x0):
        x = x0
        history = [x]
        values = [self.ackley_1d(x)]

        for _ in range(self.max_iter):
            grad = self.gradient_ackley_1d(x)
            new_x = x - self.step_size * grad

            history.append(new_x)
            values.append(self.ackley_1d(new_x))

            if np.abs(new_x - x) < self.tol:
                break

            x = new_x

        return x, self.ackley_1d(x), history, values
