from gradient_descent_ackley import GradientDescentAckley
import matplotlib.pyplot as plt
import numpy as np


def step_size_tests():
    step_sizes = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    ackley_1d_results = []
    ackley_1d_values = []

    for step_size in step_sizes:
        print(f"\nStep size = {step_size}")

        optimizer = GradientDescentAckley(step_size, max_iter=1000)
        x_output, f_output, history, values = optimizer.optimize_ackley_1d(x0=4.0)
        ackley_1d_values.append(values)
        ackley_1d_results.append({
            'step': step_size,
            "iterations": len(history),
            'optimum_val': f_output,
            "optimum_x": x_output,
        })
        print(f"  1D: x={x_output:.6f}, f(x)={f_output:.6f}, iterations={len(history)}")

    return ackley_1d_results, ackley_1d_values


def visualize_results(values_1d):
    # 1. 1D Plot
    x = np.linspace(-5, 5, 400)
    y = [-20 * np.exp(-0.2 * np.abs(xi)) - np.exp(np.cos(2 * np.pi * xi)) + 20 + np.exp(1) for xi in x]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(x, y, 'b-', linewidth=2)
    plt.title('Funkcja Ackleya 1D')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    step_sizes = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    colors = ['r', 'g', 'b', 'o', 'y', 'p']

    for step, color in zip(step_sizes, colors):
        values = values_1d[step_sizes.index(step)]
        plt.plot(range(len(values)), values, color + '-', linewidth=2, label=f'krok={step}')

    plt.title('Zbieżność dla różnych kroków (1D)')
    plt.xlabel('Iteracja')
    plt.ylabel('Wartość funkcji')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ackley_results_1d, ackley_1d_values = step_size_tests()
    print("\nDone calculating")
    visualize_results(ackley_1d_values)
