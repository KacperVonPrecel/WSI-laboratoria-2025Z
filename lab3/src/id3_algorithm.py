from solver import Solver
from node import Node
from typing import Any, Callable
import pandas as pd
import math


class DecisionSolver(Solver):

    def get_parameters(self):
        pass

    def fit(self, df: pd.DataFrame, depth: int) -> Node:
        temp_list = df.columns.tolist()
        attributes_list = temp_list[1:-1]
        target = temp_list[-1]
        return self._id3(df, attributes_list, target, depth)

    def _id3(self, df: pd.DataFrame, attributes: list, target: str, depth: int) -> Node:
        if len(df[target].unique()) == 1:
            return Node(label=df[target].iloc[0])

        if not attributes or depth == 0:
            return Node(label=df[target].mode()[0])

        gains = [self._infgain(df, attr, target) for attr in attributes]
        best_attr = attributes[gains.index(max(gains))]

        root = Node(attribute=best_attr)

        for value in df[best_attr].unique():
            subset = df[df[best_attr] == value]
            child = self._id3(subset, [attr for attr in attributes if attr != best_attr], target, depth - 1)
            root.children[value] = child

        return root

    def _entropy(self, series: pd.Series):
        counts = series.value_counts()
        total = len(series)
        ent = 0
        for count in counts:
            p = count / total
            ent -= p * math.log(p)
        return ent

    def _infgain(self, df: pd.DataFrame, attribute: str, target: str):
        total_entropy = self._entropy(df[target])
        weighted_entropy = 0

        for value in df[attribute].unique():
            series = df[df[attribute] == value]
            series_entropy = self._entropy(series)
            weighted_entropy += (len(series) / len(df)) * series_entropy

        return total_entropy - weighted_entropy

    def check_accuracy(self, df: pd.DataFrame, node: Node, target: str):
        correct = 0
        predicted_classes = self.predict(df, node)
        i = 0
        for _, row in df.iterrows():
            if predicted_classes[i] == row[target]:
                correct += 1
            i += 1
        return correct / len(df)

    def predict(self, df: pd.DataFrame, node: Node):
        predicted_classes = []
        for _, row in df.iterrows():
            predicted_classes.append(self._predict_sample(row, node))
        return predicted_classes

    def _predict_sample(self, sample, node: Node):
        if node.label is not None:
            return node.label
        curr_attr_val = sample[node.attribute]
        child = node.children.get(curr_attr_val)
        if child is None:
            return None
        return self._predict_sample(sample, child)
