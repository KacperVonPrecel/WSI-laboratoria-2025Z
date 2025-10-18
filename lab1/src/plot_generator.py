import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, Tuple


def box_plot_generation(file_handle, tested_parameter: str, plots_title: str):
    source_table = [[float(x) for x in line.split()] for line in file_handle if line.strip()]
    data_table = source_table[:-1]
    axis_x_labels = source_table[-1]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.set_xticklabels(axis_x_labels)
    ax.set_xlabel(tested_parameter, fontsize=18)
    ax.set_ylabel("Values", rotation=90, fontsize=18)

    bp = ax.boxplot(data_table, patch_artist=True, notch=False, vert=1)

    for patch in bp['boxes']:
        patch.set_facecolor("#173CB6")

    for median in bp['medians']:
        median.set(color= "#E60606", linewidth = 4)

    plt.title(plots_title, fontweight='bold', fontsize=32)
    plt.savefig(plots_title)
    plt.show()


if __name__ == "__main__":
    print("Please input a file's name from which the boxplot will be created:")
    file_name = str(input())
    print("Please enter the name of tested parameter:")
    param_name = str(input())
    print("Please enter the title of the plot:")
    title = str(input())
    with open(file_name, 'r') as f_handle:
        box_plot_generation(f_handle, param_name, title)
