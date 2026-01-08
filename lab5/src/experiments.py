import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings
from data_reader import DataReader
from layer import DenseLayer
from activation_functions import ReLU, Sigmoid, Linear
from loss_functions import SoftmaxCrossEntropy

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def train_network(X, y_onehot, config):
    name = config['name']
    hidden_dims = config['layers']
    act_name = config['act']
    lr = config['lr']
    epochs = config.get('epochs', 60)
    batch_size = 32

    print(f"--- [START] Eksperyment: {name} ---")

    input_dim = X.shape[1]
    output_dim = y_onehot.shape[1]

    if act_name == 'relu':
        HiddenAct = ReLU
    elif act_name == 'sigmoid':
        HiddenAct = Sigmoid
    else:
        HiddenAct = ReLU

    network = []

    network.append(DenseLayer(input_dim, hidden_dims[0], HiddenAct(), lr))

    for i in range(1, len(hidden_dims)):
        network.append(DenseLayer(hidden_dims[i - 1], hidden_dims[i], HiddenAct(), lr))

    network.append(DenseLayer(hidden_dims[-1], output_dim, Linear(), lr))

    loss_fn = SoftmaxCrossEntropy()

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
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

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
                    if layer.trainable:
                        layer.weights -= lr * layer.grad_weights
                        layer.biases -= lr * layer.grad_biases

            avg_loss = epoch_loss / (X.shape[0] / batch_size)
            accuracy = correct_predictions / X.shape[0]

            loss_history.append(avg_loss)
            acc_history.append(accuracy)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}")

    except KeyboardInterrupt:
        print("\n[INFO] Przerwano trening ręcznie.")

    print(f"--- [KONIEC] {name} | Final Acc: {acc_history[-1]:.2%} ---")
    return loss_history, acc_history


if __name__ == "__main__":
    filename = "wsi5-25Z_dataset.csv"

    if not os.path.exists(filename) and os.path.exists(os.path.join("..", filename)):
        filename = os.path.join("..", filename)

    if not os.path.exists(filename):
        print(f"[ERROR] Nie znaleziono pliku {filename}. Upewnij się, że jest w folderze.")
        sys.exit(1)

    print("Wczytywanie danych...")
    reader = DataReader()
    seed = 1234

    with open(filename, 'r') as f:
        reader.read_data(f, seed)

    X_train = reader.get_train_x()
    y_train = reader.get_train_y()

    print(f"Dane gotowe. Próbek: {X_train.shape[0]}, Cech: {X_train.shape[1]}, Klas: {y_train.shape[1]}")

    scenarios = [
        {"name": "1. Baseline (ReLU, LR=0.01)", "layers": [64, 32], "act": "relu", "lr": 0.01},
        {"name": "2. Sigmoid (LR=0.01)", "layers": [64, 32], "act": "sigmoid", "lr": 0.01},
        {"name": "3. Small LR (LR=0.001)", "layers": [64, 32], "act": "relu", "lr": 0.001},
        {"name": "4. High LR (LR=0.1)", "layers": [64, 32], "act": "relu", "lr": 0.1},
        {"name": "5. Shallow Net (1 warstwa)", "layers": [64], "act": "relu", "lr": 0.01},
    ]

    results = {}

    for s in scenarios:
        losses, accs = train_network(X_train, y_train, s)
        results[s["name"]] = {"loss": losses, "acc": accs}

    print("\nGenerowanie wykresów...")

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data["loss"], label=name)
    plt.title("Funkcja kosztu (Loss) w czasie treningu")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data["acc"], label=name)
    plt.title("Dokładność (Accuracy) w czasie treningu")
    plt.xlabel("Epoka")
    plt.ylabel("Accuracy (0-1)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("wyniki_eksperymentow.png")
    plt.show()
    print("\nGotowe! Wykres zapisano jako 'wyniki_eksperymentow.png'.")
