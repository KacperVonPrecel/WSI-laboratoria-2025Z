from mlp import MLP, DenseLayer
from data_reader import DataReader
from activation_functions import ReLU, Linear, Sigmoid
from loss_functions import SoftmaxCrossEntropy
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    filename = args.filename

    reader = DataReader()
    mlp = MLP()

    seed = 37

    reader.read_data(filename, seed)

    relu = ReLU()
    linear = Linear()
    loss_fn = SoftmaxCrossEntropy()

    # learnig rate in a layer is 0.01 by default
    mlp.add(DenseLayer(11, 32, activation_func=relu))
    mlp.add(DenseLayer(32, 16, activation_func=relu))
    mlp.add(DenseLayer(16, 6, activation_func=linear))

    labels_data = reader.get_train_x()
    class_data = reader.get_train_y()

    labels_val = reader.get_val_x()
    class_val = reader.get_val_y()

    labels_test = reader.get_test_x()
    class_test = reader.get_test_y()

    print("Training with evaluation; Epochs = 10000:")
    print("=" * 120)
    mlp.train(labels_data, class_data, loss_fn, labels_val, class_val, epochs=600)

    print("=" * 120)
    print("Test set:")
    print("=" * 120)
    test_loss, test_accuracy = mlp.evaluate(labels_test, class_test, loss_fn)
    print(f"Test Loss: {test_loss:.4f}, Test accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
