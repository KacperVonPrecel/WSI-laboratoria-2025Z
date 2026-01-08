import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings
from data_reader import DataReader
from layer import DenseLayer, LayerDropout
from activation_functions import ReLU, Sigmoid, Linear
from loss_functions import SoftmaxCrossEntropy
from optimiser_with_regulations import SGD, Adam


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def train_network(X, y_onehot, config):
    name = config['name']
    hidden_dims = config['layers']
    act_name = config['act']
    lr = config.get('lr', 0.01)
    epochs = config.get('epochs', 60)
    batch_size = 32
    l2_reg = config.get('l2', 0.0)
    l1_reg = config.get('l1', 0.0)
    dropout_rate = config.get('dropout', 0.0)
    decay = config.get('decay', 0.0)

    print(f"--- [START] Eksperyment: {name} ---")
    print(f"    Params: LR={lr}, L2={l2_reg}, Dropout={dropout_rate}, Decay={decay}")

    input_dim = X.shape[1]
    output_dim = y_onehot.shape[1]

    if act_name == 'relu':
        HiddenAct = ReLU
    elif act_name == 'sigmoid':
        HiddenAct = Sigmoid
    else:
        HiddenAct = ReLU


    network = []

    optimizer = SGD(learning_rate=lr, decay=decay, momentum=0.9, l1=l1_reg, l2=l2_reg)

    network.append(DenseLayer(input_dim, hidden_dims[0], HiddenAct(), 0.0))
    if dropout_rate > 0:
        network.append(LayerDropout(dropout_rate))

    for i in range(1, len(hidden_dims)):
        network.append(DenseLayer(hidden_dims[i - 1], hidden_dims[i], HiddenAct(), 0.0))
        if dropout_rate > 0:
            network.append(LayerDropout(dropout_rate))

    network.append(DenseLayer(hidden_dims[-1], output_dim, Linear(), 0.0))

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
                    if isinstance(layer, LayerDropout):
                        layer.forward(output, training=True)
                    else:
                        layer.forward(output)
                    output = layer.output

                loss_val = loss_fn.forward(output, y_batch)
                epoch_loss += loss_val

                preds = np.argmax(output, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                correct_predictions += np.sum(preds == true_labels)

                dout = loss_fn.backward()

                for layer in reversed(network):
                    dout = layer.backward(dout)

                optimizer.pre_update_params()
                for layer in network:
                    if hasattr(layer, 'weights'): 
                        optimizer.step(layer)
                optimizer.post_update_params()

            avg_loss = epoch_loss / (X.shape[0] / batch_size)
            accuracy = correct_predictions / X.shape[0]

            loss_history.append(avg_loss)
            acc_history.append(accuracy)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%} | LR: {optimizer.current_learning_rate:.5f}")

    except KeyboardInterrupt:
        print("\n[INFO] Przerwano trening ręcznie.")

    print(f"--- [KONIEC] {name} | Final Acc: {acc_history[-1]:.2%} ---")
    return loss_history, acc_history


if __name__ == "__main__":
    filename = "wsi5-25Z_dataset.csv"

    if not os.path.exists(filename) and os.path.exists(os.path.join("..", filename)):
        filename = os.path.join("..", filename)

    if not os.path.exists(filename):
        print(f"[ERROR] Nie znaleziono pliku {filename}.")
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
        {
            "name": "1. Baseline (No Reg)", 
            "layers": [128, 64], 
            "act": "relu", 
            "lr": 0.1, 
            "decay": 1e-3,
            "epochs": 200
        },

        {
            "name": "2. L2 Regularization (5e-4)",
            "layers": [128, 64],
            "act": "relu",
            "lr": 0.1,
            "decay": 1e-3,
            "l2": 5e-4,
            "epochs": 200
        },

        {
            "name": "3. Dropout (20%)",
            "layers": [128, 64],
            "act": "relu",
            "lr": 0.1,
            "decay": 1e-3,
            "dropout": 0.2,
            "epochs": 200
        },
        {
            "name": "4. L2 + Dropout",
            "layers": [128, 64],
            "act": "relu",
            "lr": 0.1,
            "decay": 1e-3,
            "l2": 5e-4,
            "dropout": 0.2,
            "epochs": 200
        },
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
    plt.title("Training Loss (z użyciem optimiser_with_regulations)")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data["acc"], label=name)
    plt.title("Training Accuracy")
    plt.xlabel("Epoka")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("porownanie_regularyzacji.png")
    plt.show()
    print("\nGotowe! Wykres zapisano jako 'porownanie_regularyzacji.png'.")