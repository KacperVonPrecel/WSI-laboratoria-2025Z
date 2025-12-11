import numpy as np
from typing import Callable


class GradientDescent:
    def __init__(self, step_size=0.1, max_iter=5000, tol=1e-8, momentum=0.9):
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.momentum = momentum

    def optimize_func(self, parameters: np.array, func_to_opt: Callable, func_grad: Callable):
        point = parameters.copy()
        history = [point.copy()]
        values = [func_to_opt(point)]

        for _ in range(self.max_iter):
            grad = func_grad(point)
            new_point = point - self.step_size * grad

            history.append(new_point.copy())
            values.append(float(func_to_opt(new_point)))

            if np.linalg.norm(new_point - point) < self.tol:
                break

            point = new_point

        return point, func_to_opt(point), history, values


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
            velocity = self.momentum * velocity - self.step_size * grad
            velocity = np.clip(velocity, -10, 10)
            new_point = point + velocity

            history.append(new_point.copy())
            values.append(float(self.ackley_2d(new_point)))

            if np.linalg.norm(new_point - point) < self.tol:
                break

            point = new_point

        return point, self.ackley_2d(point), history, values
