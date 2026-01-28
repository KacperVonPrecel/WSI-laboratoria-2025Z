from solver import Solver
import numpy as np


class NaiveBayesSolver(Solver):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.priors = {}
        self.likelihoods = {}
        self.classes = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for label in self.classes:
            X_label = X[y == label]

            self.priors[label] = X_label.shape[0] / n_samples

            word_counts = np.sum(X_label, axis=0)
            total_counts = np.sum(word_counts) + (self.alpha * n_features)

            self.likelihoods[label] = (word_counts + self.alpha) / total_counts

    def predict(self, X):
        predictions = []
        for sample in X:
            posteriors = {}
            for label in self.classes:
                prior = np.log(self.priors[label])
                posterior = prior + np.sum(sample * np.log(self.likelihoods[label]))
                posteriors[label] = posterior

            predictions.append(max(posteriors, key=posteriors.get))

        return np.array(predictions)
