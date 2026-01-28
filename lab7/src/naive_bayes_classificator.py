from solver import Solver
import numpy as np


class NaiveBayesSolver(Solver):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
