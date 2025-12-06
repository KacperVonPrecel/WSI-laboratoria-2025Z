from gradient_descent_ackley import GradientDescentAckley
import matplotlib.pyplot as plt
import numpy as np


def step_size_tests():
    step_sizes = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7]
    ackley_1d_results = []
    ackley_1d_values = []
    ackley_2d_results = []
    ackley_2d_values = []

    for step_size in step_sizes:
        print(f"\nStep size = {step_size}")
        print(f"Starting point: x = 4.0")
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

        print("Starting point: (x = 4.0, y = 4.0)")
        xy_output, f_output, history, values = optimizer.optimize_ackley_2d(x0=4.0, y0=4.0)
        ackley_2d_values.append(values)
        ackley_2d_results.append({
            'step': step_size,
            "iterations": len(history),
            'optimum_val': f_output,
            "optimum_x": xy_output,
        })
        print(f"  2D: (x={xy_output[0]:.6f}, y={xy_output[1]:.6f}) f(x, y)={f_output:.6f}, iterations={len(history)}")

    return ackley_1d_results, ackley_1d_values, ackley_2d_results, ackley_2d_values


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

    # 2. Wykres funkcji 2D
    plt.subplot(2, 2, 2, projection='3d')
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = -20*np.exp(-0.2*np.sqrt((X[i, j]**2 + Y[i, j]**2)/2)) - \
                      np.exp((np.cos(2*np.pi*X[i, j]) + np.cos(2*np.pi*Y[i, j]))/2) + 20 + np.exp(1)

    ax = plt.gca()
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_title('Funkcja Ackleya 2D')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')

    # 3. Wykres dla 1D
    plt.subplot(2, 2, 3)
    # step_sizes = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    # colors = ['r', 'g', 'b', 'o', 'y', 'p']

    step_sizes = [0.001, 0.01, 0.05]
    colors = ['r', 'g', 'b']

    for step, color in zip(step_sizes, colors):
        values = values_1d[step_sizes.index(step)]
        plt.plot(range(len(values)), values, color + '-', linewidth=2, label=f'krok={step}')

    plt.title('Zbieżność dla różnych kroków (1D)')
    plt.xlabel('Iteracja')
    plt.ylabel('Wartość funkcji')
    plt.legend()
    plt.grid(True)

    # 4. Trajektoria optymalizacji 2D
    plt.subplot(2, 2, 4)
    optimizer = GradientDescentAckley(step_size=0.1, max_iter=1000)
    x_opt, f_opt, history, values = optimizer.optimize_ackley_2d(x0=3.0, y0=4.0)
    history = np.array(history)

    # Kontury funkcji
    xx = np.linspace(-5, 5, 100)
    yy = np.linspace(-5, 5, 100)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = np.zeros_like(XX)

    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            ZZ[i, j] = optimizer.ackley_2d(XX[i, j], YY[i, j])

    plt.contour(XX, YY, ZZ, levels=50, cmap='viridis', alpha=0.5)
    plt.plot(history[:, 0], history[:, 1], 'ro-', linewidth=2, markersize=4)
    plt.plot(history[0, 0], history[0, 1], 'go', markersize=10, label='Start')
    plt.plot(history[-1, 0], history[-1, 1], 'bo', markersize=10, label='Koniec')
    plt.title('Trajektoria optymalizacji 2D')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ackley_1d_results, ackley_1d_values, ackley_2d_results, ackley_2d_values = step_size_tests()
    print("\nDone calculating")
    visualize_results(ackley_1d_values)
