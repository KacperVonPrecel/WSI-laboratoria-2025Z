import numpy as np


class AckleyFunc:
    @classmethod
    def ackley_1d(cls, parameters):
        x = parameters[0]
        return -20 * np.exp(-0.2 * np.abs(x)) - np.exp(np.cos(2 * np.pi * x)) + 20 + np.exp(1)

    @classmethod
    def gradient_ackley_1d(cls, parameters):
        x = parameters[0]
        if np.abs(x) < 1e-10:
            sign = 0
        else:
            sign = x / np.abs(x)

        return 4 * sign * np.exp(-0.2 * np.abs(x)) + 2 * np.pi * np.sin(2 * np.pi * x) * np.exp(np.cos(2 * np.pi * x))

    @classmethod
    def ackley_2d(cls, parameters):
        x = parameters[0]
        y = parameters[1]
        return -20 * np.exp(-0.2 * np.sqrt((x**2 + y**2) / 2)) - np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2) + 20 + np.exp(1)

    @classmethod
    def gradient_ackley_2d(cls, parameters):
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
