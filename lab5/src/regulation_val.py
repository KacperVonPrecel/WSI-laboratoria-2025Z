import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings
from data_reader import DataReader
from layer import DenseLayer, LayerDropout
from activation_functions import ReLU, Sigmoid, Linear
from loss_functions import SoftmaxCrossEntropy
from optimiser_with_regulations import SGD


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def train_network(X_train, y_train, X_val, y_val, config):
    """
    Funkcja trenuje sieć, ale zwraca historię wyników na zbiorze WALIDACYJNYM.
    """
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

    print(f"--- [START] {name} ---")

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

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

    val_loss_history = []
    val_acc_history = []

    try:
        for epoch in range(epochs):

            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = X_batch
                for layer in network:
                    if isinstance(layer, LayerDropout):
                        layer.forward(output, training=True)
                    else:
                        layer.forward(output)
                    output = layer.output

                loss_fn.forward(output, y_batch)
                dout = loss_fn.backward()

                for layer in reversed(network):
                    dout = layer.backward(dout)

                optimizer.pre_update_params()
                for layer in network:
                    if hasattr(layer, 'weights'):
                        optimizer.step(layer)
                optimizer.post_update_params()

            output_val = X_val
            for layer in network:
                if isinstance(layer, LayerDropout):
                    layer.forward(output_val, training=False)
                else:
                    layer.forward(output_val)
                output_val = layer.output

            loss_val = loss_fn.forward(output_val, y_val)

            preds = np.argmax(output_val, axis=1)
            true_labels = np.argmax(y_val, axis=1)
            acc_val = np.mean(preds == true_labels)

            val_loss_history.append(loss_val)
            val_acc_history.append(acc_val)

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Val Acc: {acc_val:.2%} | Val Loss: {loss_val:.4f}")

    except KeyboardInterrupt:
        print("\n[INFO] Przerwano trening ręcznie.")

    print(f"--- [KONIEC] {name} | Final Val Acc: {val_acc_history[-1]:.2%} ---")
    return val_loss_history, val_acc_history


if __name__ == "__main__":
    filename = "wsi5-25Z_dataset.csv"
    if not os.path.exists(filename):
        if os.path.exists(os.path.join("..", filename)):
            filename = os.path.join("..", filename)
        else:
            print(f"[ERROR] Nie znaleziono pliku {filename}")
            sys.exit(1)

    print("Wczytywanie danych...")
    reader = DataReader()
    seed = 1234

    with open(filename, 'r') as f:
        reader.read_data(f, seed)

    X_train = reader.get_train_x()
    y_train = reader.get_train_y()

    X_val = reader.get_test_x()
    y_val = reader.get_test_y()

    print(f"Dane gotowe. Trening: {X_train.shape[0]}, Walidacja: {X_val.shape[0]}")

    scenarios = [
        {
            "name": "1. Baseline (Brak Reg)", 
            "layers": [128, 64], "act": "relu", "lr": 0.1, "decay": 1e-3,
            "epochs": 200
        },
        {
            "name": "2. L2 Regularization", 
            "layers": [128, 64], "act": "relu", "lr": 0.1, "decay": 1e-3,
            "l2": 5e-4, "epochs": 200
        },
        {
            "name": "3. Dropout (20%)", 
            "layers": [128, 64], "act": "relu", "lr": 0.1, "decay": 1e-3,
            "dropout": 0.2, "epochs": 200
        },
        {
            "name": "4. L2 + Dropout",
            "layers": [128, 64], "act": "relu", "lr": 0.1, "decay": 1e-3,
            "l2": 5e-4, "dropout": 0.2, "epochs": 200
        },
    ]

    results = {}

    for s in scenarios:
        loss_hist, acc_hist = train_network(X_train, y_train, X_val, y_val, s)
        results[s["name"]] = {"loss": loss_hist, "acc": acc_hist}

    print("\nGenerowanie wykresu walidacyjnego...")

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data["loss"], label=name, linewidth=2)
    plt.title("Walidacja: Funkcja Kosztu (Validation Loss)")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data["acc"], label=name, linewidth=2)
    plt.title("Walidacja: Dokładność (Validation Accuracy)")
    plt.xlabel("Epoka")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("wykres_regulacji_zbiór_walidacyjny.png")
    plt.show()
    print("\nGotowe! Zapisano 'wykres_walidacyjny.png'.")