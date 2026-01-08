import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

try:
    from data_reader import DataReader
    from layer import DenseLayer
    from activation_functions import ReLU, Sigmoid, Linear
    from loss_functions import SoftmaxCrossEntropy
    from optimiser import SGD, Adam
except ImportError as e:
    print(f"[CRITICAL ERROR] Brakuje pliku lub klasy: {e}")
    sys.exit(1)


def train_network(X, y_onehot, config):
    name = config['name']
    hidden_dims = config['layers']
    opt_name = config['optimiser']
    lr = config['lr']
    epochs = 50
    batch_size = 32

    print(f"--- [START] Eksperyment: {name} ---")

    input_dim = X.shape[1]
    output_dim = y_onehot.shape[1]

    network = []
    network.append(DenseLayer(input_dim, hidden_dims[0], ReLU(), lr))
    for i in range(1, len(hidden_dims)):
        network.append(DenseLayer(hidden_dims[i-1], hidden_dims[i], ReLU(), lr))
    network.append(DenseLayer(hidden_dims[-1], output_dim, Linear(), lr))

    loss_fn = SoftmaxCrossEntropy()

    if opt_name == 'adam':
        optimizer = Adam(learning_rate=lr)
    else:
        optimizer = SGD(learning_rate=lr)

    loss_history = []
    acc_history = []

    try:
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]

            epoch_loss = 0
            correct_predictions = 0

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                output = X_batch
                for layer in network:
                    output = layer.forward(output)

                loss_val = loss_fn.forward(output, y_batch)
                epoch_loss += loss_val

                preds = np.argmax(output, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                correct_predictions += np.sum(preds == true_labels)

                dout = loss_fn.backward()
                for layer in reversed(network):
                    dout = layer.backward(dout)

                for layer in network:
                    optimizer.step(layer)

            avg_loss = epoch_loss / (X.shape[0] / batch_size)
            accuracy = correct_predictions / X.shape[0]
            loss_history.append(avg_loss)
            acc_history.append(accuracy)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}")

    except KeyboardInterrupt:
        print("\nPrzerwano.")

    print(f"--- [KONIEC] {name} | Final Acc: {acc_history[-1]:.2%} ---")
    return loss_history, acc_history


if __name__ == "__main__":
    filename = "wsi5-25Z_dataset.csv"
    if not os.path.exists(filename) and os.path.exists(os.path.join("..", filename)):
        filename = os.path.join("..", filename)

    reader = DataReader()
    with open(filename, 'r') as f:
        reader.read_data(f, 1234)

    X_train = reader.get_train_x()
    y_train = reader.get_train_y()

    scenarios = [
        {"name": "1. SGD (LR=0.01)",   "layers": [64, 32], "optimiser": "sgd",  "lr": 0.01},
        {"name": "2. Adam (LR=0.001)", "layers": [64, 32], "optimiser": "adam", "lr": 0.001},
        {"name": "3. Adam (LR=0.01)",  "layers": [64, 32], "optimiser": "adam", "lr": 0.01},
    ]

    results = {}
    for s in scenarios:
        losses, accs = train_network(X_train, y_train, s)
        results[s["name"]] = {"loss": losses, "acc": accs}

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data["loss"], label=name)
    plt.title("SGD vs Adam: Funkcja kosztu (Loss)")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data["acc"], label=name)
    plt.title("SGD vs Adam: Dokładność (Accuracy)")
    plt.xlabel("Epoka")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig("porownanie_adam_sgd.png")
    plt.show()
    print("Zapisano wykres jako porownanie_adam_sgd.png")