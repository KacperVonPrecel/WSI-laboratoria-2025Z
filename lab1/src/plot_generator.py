import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, Tuple


def box_plot_generation(file_handle):
    source_table = [[float(x) for x in line.split()] for line in file_handle if line.strip()]
    data_table = source_table[:-1]
    axis_x_labels = source_table[-1]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.set_xticklabels(axis_x_labels)
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()

    bp = ax.boxplot(data_table)

    plt.title("Testowy boxplot")
    plt.savefig("Testowy boxplot")
    plt.show()


if __name__ == "__main__":
    with open('test_boxplot_data.txt', 'r') as f_handle:
        box_plot_generation(f_handle)
