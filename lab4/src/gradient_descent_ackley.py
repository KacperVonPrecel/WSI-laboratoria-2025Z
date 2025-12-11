import numpy as np


class GradientDescentAckley:
    def __init__(self, step_size=0.1, max_iter=5000, tol=1e-8, momentum=0.9):
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.momentum = momentum

    def ackley_1d(self, parameters):
        x = parameters[0]
        return -20 * np.exp(-0.2 * np.abs(x)) - np.exp(np.cos(2 * np.pi * x)) + 20 + np.exp(1)

    def gradient_ackley_1d(self, parameters):
        x = parameters
        if np.abs(x) < 1e-10:
            sign = 0
        else:
            sign = x / np.abs(x)

        return 4 * sign * np.exp(-0.2 * np.abs(x)) + 2 * np.pi * np.sin(2 * np.pi * x) * np.exp(np.cos(2 * np.pi * x))

    def optimize_ackley_1d(self, parameters):
        x = np.array(parameters)
        history = [x]
        values = [self.ackley_1d(x)]

        for _ in range(self.max_iter):
            grad = self.gradient_ackley_1d(x)
            new_x = x - self.step_size * grad

            history.append(new_x)
            values.append(float(self.ackley_1d(new_x)))

            if np.abs(new_x - x) < self.tol:
                break

            x = new_x

        return x[0], self.ackley_1d(x), history, values

    def ackley_2d(self, parameters):
        x = parameters[0]
        y = parameters[1]
        return -20 * np.exp(-0.2 * np.sqrt((x**2 + y**2) / 2)) - np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2) + 20 + np.exp(1)

    def gradient_ackley_2d(self, parameters):
        x = parameters[0]
        y = parameters[1]
        norm = np.sqrt(x**2 + y**2)
        if norm < 1e-10:
            term1_x = 0
            term1_y = 0
        else:
            term1_x = (2 * x / norm) * np.exp(-0.2 * norm / np.sqrt(2))
            term1_y = (2 * y / norm) * np.exp(-0.2 * norm / np.sqrt(2))

        exp_term = np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi* y))/2)
        term2_x = np.pi * np.sin(2 * np.pi * x) * exp_term
        term2_y = np.pi * np.sin(2 * np.pi * y) * exp_term

        return np.array([term1_x + term2_x, term1_y + term2_y])

    def optimize_ackley_2d(self, parameters):
        point = np.array(parameters)
        history = [point.copy()]
        values = [self.ackley_2d(point)]

        for _ in range(self.max_iter):
            grad = self.gradient_ackley_2d(point)
            new_point = point - self.step_size * grad

            history.append(new_point.copy())
            values.append(float(self.ackley_2d(new_point)))

            if np.linalg.norm(new_point - point) < self.tol:
                break

            point = new_point

        return point, self.ackley_2d(point), history, values

    def optimize_ackley_1d_sgd_and_mom(self, parameters):
        x = np.array(parameters)
        vx = 0.0
        history = [x]
        values = [self.ackley_1d(x)]

        for _ in range(self.max_iter):
            noise = np.random.normal(0, 0.05, 1)
            grad = self.gradient_ackley_1d(x) + noise
            vx = self.momentum * vx - self.step_size * grad
            vx = np.clip(vx, -10, 10)
            new_x = x + vx

            history.append(new_x)
            values.append(float(self.ackley_1d(new_x)))

            if np.abs(new_x - x) < self.tol:
                break

            x = new_x

        return float(x[0]), float(self.ackley_1d(x)), history, values

    def optimize_ackley_2d_sgd_and_mom(self, parameters):
        point = np.array(parameters)
        velocity = np.zeros(2)
        history = [point.copy()]
        values = [self.ackley_2d(point)]

        for _ in range(self.max_iter):
            noise = np.random.normal(0, 0.05, 2)
            grad = self.gradient_ackley_2d(point) + noise
            velocity[0] = self.momentum * velocity[0] - self.step_size * grad[0]
            velocity[1] = self.momentum * velocity[1] - self.step_size * grad[1]
            velocity = np.clip(velocity, -10, 10)
            new_point = point + velocity

            history.append(new_point.copy())
            values.append(float(self.ackley_2d(new_point)))

            if np.linalg.norm(new_point - point) < self.tol:
                break

            point = new_point

        return point, self.ackley_2d(point), history, values
