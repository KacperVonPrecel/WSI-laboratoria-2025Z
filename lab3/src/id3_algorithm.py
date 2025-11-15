from solver import Solver
from node import Node
from typing import Any, Callable
import pandas as pd
import math


class DecisionSolver(Solver):

    def get_parameters(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def id3(self, df: pd.DataFrame, attributes: list, target: str, depth: int):
        if len(df[target].unique()) == 1:
            return Node(label=df[target].iloc[0])

        if not attributes or depth == 0:
            return Node(label=df[target].mode()[0])

        gains = [self.infgain(df, attr, target) for attr in attributes]
        best_attr = attributes[gains.index(max(gains))]

        root = Node(attribute=best_attr)

        for value in df[best_attr].unique():
            subset = df[df[best_attr] == value]
            child = self.id3(subset, [attr for attr in attributes if attr != best_attr], target, depth - 1)
            root.children[value] = child

        return root

    def entropy(self, series: pd.Series):
        counts = series.value_counts()
        total = len(series)
        ent = 0
        for count in counts:
            p = count / total
            ent -= p * math.log(p)
        return ent

    def infgain(self, df: pd.DataFrame, attribute: str, target: str):
        total_entropy = self.entropy(df[target])
        weighted_entropy = 0

        for value in df[attribute].unique():
            series = df[df[attribute] == value]
            series_entropy = self.entropy(series)
            weighted_entropy += (len(series) / len(df)) * series_entropy

        return total_entropy - weighted_entropy
