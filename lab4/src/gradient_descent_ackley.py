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

    def ackley_2d(self, x, y):
        return -20 * np.exp(-0.2 * np.sqrt((x**2 + y**2) / 2)) - np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2) + 20 + np.exp(1)

    def gradient_ackley_2d(self, x, y):
        norm = np.sqrt(x**2 + y**2)
        if norm < 1e-10:
            term1_x = 0
            term2_y = 0
        else:
            term1_x = (2 * x / norm) * np.exp(-0.2 * norm / np.sqrt(2))
            term1_y = (2 * y / norm) * np.exp(-0.2 * norm / np.sqrt(2))

        exp_term = np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi* y))/2)
        term2_x = np.pi * np.sin(2 * np.pi * x) * exp_term
        term2_y = np.pi * np.sin(2 * np.pi * y) * exp_term

        return np.array([term1_x + term2_x, term1_y + term2_y])

    def optimize_ackley_2d(self, x0, y0):
        point = np.array([x0, y0])
        history = [point.copy()]
        values = [self.ackley_2d(point[0], point[1])]

        for _ in range(self.max_iter):
            grad = self.gradient_ackley_2d(point[0], point[1])
            new_point = point - self.step_size * grad

            history.append(new_point.copy())
            values.append(self.ackley_2d(new_point[0], new_point[1]))

            if np.linalg.norm(new_point - point) < self.tol:
                break

            point = new_point

        return point, self.ackley_2d(point[0], point[1]), history, values
