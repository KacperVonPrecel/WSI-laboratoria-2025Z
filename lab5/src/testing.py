from mlp import MLP, DenseLayer
from data_reader import DataReader, get_class_values, get_labels_values
from activation_functions import ReLU, Linear, Sigmoid
import argparse
import random


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('filename')
    # args = parser.parse_args()
    # filename = args.filename

    reader = DataReader()
    mlp = MLP()

    seed = 37

    reader.read_data("./wsi5-25Z_dataset.csv", seed)

    train_df = reader.get_train_df()
    relu = ReLU()
    linear = Linear()

    mlp.add(DenseLayer(11, 32, activation_func=relu))
    mlp.add(DenseLayer(32, 16, activation_func=relu))
    mlp.add(DenseLayer(16, 6, activation_func=linear))

    labels_data = get_labels_values(train_df)
    class_data = get_class_values(train_df)

    print("Training set: \n")
    print("=" * 120)
    mlp.train(labels_data, class_data, epochs=10000)

    print("=" * 120)
    print("Validation set: \n")
    print("=" * 120)
    mlp.train(labels_data, class_data, epochs=10000)

if __name__ == "__main__":
    main()
