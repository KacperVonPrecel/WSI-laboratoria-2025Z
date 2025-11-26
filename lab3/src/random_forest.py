from id3_algorithm import DecisionSolver
from node import Node
import pandas as pd
import math
import random
from collections import Counter

DEPTH_NULLIFIER = 100


class RandomForest(DecisionSolver):

    def choose_random_attributes(self, attributes, k):
        return random.sample(attributes, k)

    def fit(self, df: pd.DataFrame, trees_num: int):
        forest = []
        temp_list = df.columns.tolist()
        attributes_list = temp_list[1:-1]
        attributes_k = int(math.sqrt(len(attributes_list)))
        target = temp_list[-1]

        for _ in range(trees_num):
            bootstrap_df = df.sample(frac=1.0, replace=True, random_state=42)
            tree = super()._id3(
                bootstrap_df,
                self.choose_random_attributes(attributes_list, attributes_k),
                target,
                DEPTH_NULLIFIER)
            forest.append(tree)

        return forest

    def predict_forest(self, df: pd.DataFrame, forest: list):
        predicted_outcomes = []
        for _, row in df.iterrows():
            sample_predictions = []
            for pos in range(len(forest)):
                sample_predictions.append(super()._predict_sample(row, forest[pos]))
            predicted_outcomes.append(Counter(sample_predictions).most_common(1)[0][0])
        return predicted_outcomes

    def check_accuracy(self, df: pd.DataFrame, forest: list, target: str):
        correct = 0
        predicted_classes = self.predict_forest(df, forest)
        i = 0
        for _, row in df.iterrows():
            if predicted_classes[i] == row[target]:
                correct += 1
            i += 1
        return correct / len(df)
