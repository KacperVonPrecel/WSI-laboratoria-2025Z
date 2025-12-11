from gradient_descent_ackley import GradientDescentAckley
import matplotlib.pyplot as plt
import numpy as np

STEP_SIZES = [0.001, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.5]
RANDOM_SEEDS = [30, 40, 50, 60, 70]

def step_size_tests():
    ackley_1d_results = []
    ackley_1d_values = []
    ackley_2d_results = []

    for step_size in STEP_SIZES:
        print(f"\nStep size = {step_size}")
        print("Starting point: x = 5.0")
        optimizer = GradientDescentAckley(step_size, max_iter=1000, tol=1e-8, momentum=0.9)
        x_output, f_output, history, values = optimizer.optimize_ackley_1d(parameters=[5.0])
        ackley_1d_values.append(values)
        ackley_1d_results.append({
            'step': step_size,
            "iterations": len(history),
            'optimum_val': f_output,
            "optimum_x": x_output,
        })
        print(f"  1D: x={x_output:.6f}, f(x)={f_output:.6f}, iterations={len(history)}")

        print("Starting point: (x = 5.0, y = 5.0)")
        xy_output, f_output, history, values = optimizer.optimize_ackley_2d(parameters=[5.0, 5.0])
        ackley_2d_results.append({
            'step': step_size,
            "iterations": len(history),
            'optimum_val': f_output,
            "optimum_xy": xy_output,
        })
        print(f"  2D: (x={xy_output[0]:.6f}, y={xy_output[1]:.6f}) f(x, y)={f_output:.6f}, iterations={len(history)}")

    return ackley_1d_results, ackley_1d_values, ackley_2d_results


def step_size_sgd_tests():
    avg_1d_output = []
    avg_1d_f_output = []
    avg_2d_x_output = []
    avg_2d_y_output = []
    avg_2d_f_output = []

    ackley_1d_sgd_results = []
    ackley_1d_sgd_values = []
    ackley_2d_sgd_results = []

    for step_size in STEP_SIZES:
        x_output_list = []
        f_1d_output_list = []
        x_only_output_list = []
        y_only_output_list = []
        f_2d_output_list = []

        for rand_seed in RANDOM_SEEDS:
            np.random.seed(rand_seed)
            print(f"\nStep size = {step_size}")
            print("Starting point: x = 5.0")
            optimizer = GradientDescentAckley(step_size, max_iter=1000, tol=1e-4, momentum=0.7)
            x_output, f_output, history, values = optimizer.optimize_ackley_1d_sgd_and_mom(parameters=[5.0])
            x_output_list.append(x_output)
            f_1d_output_list.append(f_output)
            ackley_1d_sgd_values.append(values)
            ackley_1d_sgd_results.append({
                'step': step_size,
                "iterations": len(history),
                'optimum_val': f_output,
                "optimum_x": x_output,
            })
            print(f"  1D: x={x_output:.6f}, f(x)={f_output:.6f}, iterations={len(history)}")

            print("Starting point: (x = 5.0, y = 5.0)")
            xy_output, f_output, history, values = optimizer.optimize_ackley_2d_sgd_and_mom(parameters=[5.0, 5.0])
            x_only_output_list.append(xy_output[0])
            y_only_output_list.append(xy_output[1])
            f_2d_output_list.append(f_output)
            ackley_2d_sgd_results.append({
                'step': step_size,
                "iterations": len(history),
                'optimum_val': f_output,
                "optimum_xy": xy_output,
            })
            print(f"  2D: (x={xy_output[0]:.6f}, y={xy_output[1]:.6f}) f(x, y)={f_output:.6f}, iterations={len(history)}")

        avg_1d_output.append(np.average(x_output_list))
        avg_1d_f_output.append(np.average(f_1d_output_list))
        avg_2d_x_output.append(np.average(x_only_output_list))
        avg_2d_y_output.append(np.average(y_only_output_list))
        avg_2d_f_output.append(np.average(f_2d_output_list))

    print("=" * 100)
    print(f"Avg x: {avg_1d_output}")
    print(f"Avg f(x): {avg_1d_f_output}")
    print("=" * 100)
    print(f"Avg x: {avg_2d_x_output}")
    print(f"Avg y: {avg_2d_y_output}")
    print(f"Avg f(x,y): {avg_2d_f_output}")
    print("=" * 100)

    return ackley_1d_sgd_results, ackley_1d_sgd_values, ackley_2d_sgd_results


def visualize_1d_results(values_1d, step_size):
    plt.subplot()
    color = 'b'

    plt.plot(range(len(values_1d)), values_1d, color + '-', linewidth=2, label=f'step={step_size}')

    plt.title(f'Zbieżność dla kroku {step_size} (1D)')
    plt.xlabel('Iteracja')
    plt.ylabel('Wartość funkcji')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_2d_results(history, step_size):
    plt.subplot(1, 1, 1)
    optimizer = GradientDescentAckley(step_size=step_size, max_iter=1000)
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
    plt.title(f'Trajektoria optymalizacji 2D: Skok={step_size}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # step_sizes = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]

    print("Gradient descent")
    ackley_1d_results, ackley_1d_values, ackley_2d_results = step_size_tests()
    print("\nDone calculating")
    # print("=" * 100)
    print("SGD with momentum")
    ackley_1d_sgd_results, ackley_1d_sgd_values, ackley_2d_sgd_results = step_size_sgd_tests()
    print("\nDone calculating")

    # for step_size in step_sizes:
    #     visualize_1d_results(values_1d=ackley_1d_values[step_sizes.index(step_size)], step_size=step_size)
    #     print("Done")

    # print("=" * 100)
    # for step_size in step_sizes:
    #     visualize_1d_results(values_1d=ackley_1d_sgd_values[step_sizes.index(step_size)], step_size=step_size)
    #     print("Done")
