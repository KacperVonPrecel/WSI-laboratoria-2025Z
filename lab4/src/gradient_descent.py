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

    def optimize_func_sgd_and_mom(self, parameters: np.array, func_to_opt: Callable, func_grad: Callable):
        point = parameters.copy()
        velocity = np.zeros(2)
        history = [point.copy()]
        values = [func_to_opt(point)]

        for _ in range(self.max_iter):
            noise = np.random.normal(0, 0.05, 2)
            grad = func_grad(point) + noise
            velocity = self.momentum * velocity - self.step_size * grad
            velocity = np.clip(velocity, -10, 10)
            new_point = point + velocity

            history.append(new_point.copy())
            values.append(float(func_to_opt(new_point)))

            if np.linalg.norm(new_point - point) < self.tol:
                break

            point = new_point

        return point, func_to_opt(point), history, values
