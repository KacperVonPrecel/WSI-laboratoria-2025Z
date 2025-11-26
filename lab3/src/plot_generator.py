import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, Tuple


def plot_generation(file_handle, plots_title: str):
    source_table = [[float(x) for x in line.split()] for line in file_handle if line.strip()]
    data_table = source_table[:-1]
    accuracy_list = np.zeros(len(data_table[0]), np.float64)
    for col in range(len(data_table[0])):
        col_list = np.zeros(len(data_table), np.float64)
        for row in range(len(data_table)):
            col_list[row] = data_table[row][col]
        accuracy_list[col] = col_list.mean()

    axis_x_labels_float = source_table[-1]
    axis_x_labels = []
    for i in range(len(axis_x_labels_float)):
        axis_x_labels.append(int(axis_x_labels_float[i]))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    x_positions = list(range(len(accuracy_list)))
    ax.set_xticklabels(axis_x_labels)
    ax.set_xticks(x_positions)

    ax.set_xlabel("Tree depth", fontsize=18)
    ax.set_ylabel("Avg accuracy in %", rotation=90, fontsize=18)

    tp = ax.bar(x_positions, accuracy_list, color="blue")

    plt.title(plots_title, fontweight='bold', fontsize=26)
    plt.savefig(plots_title)
    plt.show()


if __name__ == "__main__":
    print("Please input a file's name from which the barplot will be created:")
    file_name = str(input())
    print("Please enter the title of the plot:")
    title = str(input())
    with open(file_name, 'r') as f_handle:
        plot_generation(f_handle, title)
