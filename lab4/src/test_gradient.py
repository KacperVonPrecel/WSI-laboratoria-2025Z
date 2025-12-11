from gradient_descent import GradientDescent
from ackley_func import AckleyFunc
import matplotlib.pyplot as plt
import numpy as np

STEP_SIZES = [0.001, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.5]
RANDOM_SEEDS = [30, 40, 50, 60, 70]
ITERATIONS = 1000


def step_size_tests():
    ackley_1d_values = []
    ackley_1d_histories = []

    ackley_2d_values = []
    ackley_2d_histories = []

    for step_size in STEP_SIZES:
        print(f"\nStep size = {step_size}")
        print("Starting point: x = 5.0")

        optimizer = GradientDescent(step_size, max_iter=1000, tol=1e-8, momentum=0.9)

        x_output, f_output, history, values = optimizer.optimize_func(
                                                        parameters=[5.0],
                                                        func_to_opt=AckleyFunc.ackley_1d,
                                                        func_grad=AckleyFunc.gradient_ackley_1d
                                                        )
        ackley_1d_values.append(values)
        ackley_1d_histories.append(history)
        print(f"  1D: x={x_output[0]:.6f}, f(x)={f_output:.6f}, iterations={len(history)}")

        print("Starting point: (x = 5.0, y = 5.0)")
        xy_output, f_output, history, values = optimizer.optimize_func(
                                                        parameters=[5.0, 5.0],
                                                        func_to_opt=AckleyFunc.ackley_1d,
                                                        func_grad=AckleyFunc.gradient_ackley_2d
                                                        )
        print(f"  2D: (x={xy_output[0]:.6f}, y={xy_output[1]:.6f}) f(x, y)={f_output:.6f}, iterations={len(history)}")
        ackley_2d_values.append(values)
        ackley_2d_histories.append(history)

    return ackley_1d_histories, ackley_1d_values, ackley_2d_histories, ackley_2d_values


def step_size_sgd_tests():
    avg_1d_output = []
    avg_1d_f_output = []
    avg_2d_x_output = []
    avg_2d_y_output = []
    avg_2d_f_output = []

    std_1d = []
    std_1d_f = []
    std_2d_x = []
    std_2d_y = []
    std_2d_f = []

    ackley_1d_sgd_history = np.empty(shape=(len(STEP_SIZES), len(RANDOM_SEEDS)), dtype=object)
    ackley_1d_sgd_values = np.empty(shape=(len(STEP_SIZES), len(RANDOM_SEEDS)), dtype=object)
    ackley_2d_sgd_history = np.empty(shape=(len(STEP_SIZES), len(RANDOM_SEEDS)), dtype=object)
    ackley_2d_sgd_values = np.empty(shape=(len(STEP_SIZES), len(RANDOM_SEEDS)), dtype=object)

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
            optimizer = GradientDescent(step_size, max_iter=ITERATIONS, tol=1e-8, momentum=0.9)
            x_output, f_output, history, values = optimizer.optimize_func_sgd_and_mom(
                                                            parameters=[5.0],
                                                            func_to_opt=AckleyFunc.ackley_1d,
                                                            func_grad=AckleyFunc.gradient_ackley_1d
                                                            )
            x_output_list.append(x_output[0])
            f_1d_output_list.append(f_output)
            ackley_1d_sgd_values[STEP_SIZES.index(step_size)][RANDOM_SEEDS.index(rand_seed)] = values
            ackley_1d_sgd_history[STEP_SIZES.index(step_size)][RANDOM_SEEDS.index(rand_seed)] = history
            print(f"  1D: x={x_output[0]:.6f}, f(x)={f_output:.6f}, iterations={len(history)}")

            print("Starting point: (x = 5.0, y = 5.0)")
            xy_output, f_output, history, values = optimizer.optimize_func_sgd_and_mom(
                                                            parameters=[5.0, 5.0],
                                                            func_to_opt=AckleyFunc.ackley_2d,
                                                            func_grad=AckleyFunc.gradient_ackley_2d
                                                            )
            x_only_output_list.append(xy_output[0])
            y_only_output_list.append(xy_output[1])
            f_2d_output_list.append(f_output)
            ackley_2d_sgd_values[STEP_SIZES.index(step_size)][RANDOM_SEEDS.index(rand_seed)] = values
            ackley_2d_sgd_history[STEP_SIZES.index(step_size)][RANDOM_SEEDS.index(rand_seed)] = history
            print(f"  2D: (x={xy_output[0]:.6f}, y={xy_output[1]:.6f}) f(x, y)={f_output:.6f}, iterations={len(history)}")

        avg_1d_output.append(np.average(x_output_list))
        std_1d.append(np.std(x_output_list))
        avg_1d_f_output.append(np.average(f_1d_output_list))
        std_1d_f.append(np.std(f_1d_output_list))

        avg_2d_x_output.append(np.average(x_only_output_list))
        std_2d_x.append(np.std(x_only_output_list))

        avg_2d_y_output.append(np.average(y_only_output_list))
        std_2d_y.append(np.std(y_only_output_list))

        std_2d_f.append(np.std(f_2d_output_list))
        avg_2d_f_output.append(np.average(f_2d_output_list))

    print("=" * 100)
    print(f"Avg x: {avg_1d_output}")
    print(f"Std x: {std_1d}")
    print(f"Avg f(x): {avg_1d_f_output}")
    print(f"Std f(x): {std_1d_f}")
    print("=" * 100)
    print(f"Avg x: {avg_2d_x_output}")
    print(f"Std x: {std_2d_x}")
    print(f"Avg y: {avg_2d_y_output}")
    print(f"Std y: {std_2d_y}")
    print(f"Avg f(x,y): {avg_2d_f_output}")
    print(f"Std f(x,y): {std_2d_f}")
    print("=" * 100)

    return ackley_1d_sgd_history, ackley_1d_sgd_values, ackley_2d_sgd_history, ackley_2d_sgd_values


def visualize_1d_results(x_history, y_history, step_size, sgd=""):
    xs = np.linspace(-7, 7, 1000)
    ys = AckleyFunc.ackley_1d([xs])

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, label="Ackley Function")

    # add your gradient-descent samples
    plt.scatter(x_history[0:250], y_history[0:250], s=12, label="First Stage", color='green')
    plt.scatter(x_history[251:500], y_history[251:500], s=12, label="Second Stage", color='yellow')
    plt.scatter(x_history[501:750], y_history[501:750], s=12, label="Third Stage", color='orange')
    plt.scatter(x_history[751:798], y_history[751:798], s=12, label="Fourth Stage", color='red')

    plt.scatter(x_history[0], y_history[0], s=100, color='brown', marker='o', label="Start Point", edgecolors='black', linewidth=2)

    plt.scatter(x_history[-1], y_history[-1], s=100, color='purple', marker='s', label="End Point", edgecolors='black', linewidth=2)

    plt.xlabel("x")
    plt.ylabel("Ackley(x)")
    plt.title(f"Gradient Descent History on 1D Ackley Function With Step {step_size}")
    plt.legend()
    plt.savefig(f'gradient_descent_1d_step_{step_size}_{sgd}.png')
    plt.show()


def visualize_2d_results(x_history, y_history, step_size):
    # Create grid for contour plot
    x = np.linspace(-7, 7, 100)
    y = np.linspace(-7, 7, 100)
    X, Y = np.meshgrid(x, y)
    Z = AckleyFunc.ackley_2d([X, Y])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Contour plot
    contours = ax.contour(X, Y, Z, 20, colors='black', alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=8)

    # Filled contour (colormap)
    contourf = ax.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.7)
    plt.colorbar(contourf, ax=ax, label='Ackley Value')

    # Plot gradient descent path
    x_history = np.array(x_history)
    y_history = np.array(y_history)

    # Plot the path with arrows
    ax.plot(x_history[:, 0], x_history[:, 1], 'r-', linewidth=2, label='GD Path')
    ax.scatter(x_history[:, 0], x_history[:, 1], c='red', s=30, alpha=0.7)

    # Highlight start and end points
    ax.scatter(x_history[0, 0], x_history[0, 1], s=200, color='green',
               marker='o', label='Start', edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(x_history[-1, 0], x_history[-1, 1], s=200, color='blue',
               marker='s', label='End', edgecolors='black', linewidth=2, zorder=5)

    # Add arrows to show direction (optional)
    for i in range(len(x_history)-1):
        ax.annotate('', xy=(x_history[i+1, 0], x_history[i+1, 1]),
                    xytext=(x_history[i, 0], x_history[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1, alpha=0.5))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'2D Ackley Function - Gradient Descent (Step size: {step_size})')
    ax.set_xlim([-7, 7])
    ax.set_ylim([-7, 7])
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'ackley_2d_gd_step_{step_size:.3f}.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    # print("Gradient descent")
    # ackley_1d_histories, ackley_1d_values, ackley_2d_histories, ackley_2d_values = step_size_tests()
    # print("\nDone calculating")
    # print("=" * 100)
    print("SGD with momentum")
    ackley_1d_sgd_histories, ackley_1d_sgd_values, ackley_2d_sgd_histories, ackley_2d_sgd_values = step_size_sgd_tests()
    print("\nDone calculating")

    # for step_size in STEP_SIZES:
    #     visualize_1d_results(ackley_1d_histories[STEP_SIZES.index(step_size)], ackley_1d_values[STEP_SIZES.index(step_size)], step_size)
    #     print("Done")

    # print("=" * 100)
    # for step_size in STEP_SIZES:
    #     visualize_2d_results(ackley_2d_histories[STEP_SIZES.index(step_size)], ackley_2d_values[STEP_SIZES.index(step_size)], step_size)
    #     print("Done")

    # for step_size in STEP_SIZES:
    #     visualize_1d_results(ackley_1d_sgd_histories[STEP_SIZES.index(step_size)][2], ackley_1d_sgd_values[STEP_SIZES.index(step_size)][2], step_size, sgd="sgd")
    #     print("Done")

    for step_size in STEP_SIZES:
        visualize_2d_results(ackley_2d_sgd_histories[STEP_SIZES.index(step_size)][2], ackley_2d_sgd_values[STEP_SIZES.index(step_size)][2], step_size)
        print("Done")
