import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, Tuple


def box_plot_generation(file_handle):
    for line in file_handle:
        source_table = line.strip().split()
    data_table = np.array(source_table[:-1], dtype=np.float64)
    axis_x_labels = source_table[-1]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_table)

    ax.set_xticklabels(axis_x_labels)
    plt.title("Testowy boxplot")

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.show()


if __name__ == "__main__":
    with open('test_boxplot.txt', 'r') as f_handle:
        box_plot_generation(f_handle)
