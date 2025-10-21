import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, Tuple
import genetic_solver as gen_sol
from calc_target import calc_target


def found_optimum_in_time_plot(solver: gen_sol.GeneticSolver, pop_size: int, plots_title: str) -> str:
    test_population = gen_sol.create_first_population(pop_size)
    _, best_value, values_history = solver.solve(calc_target, test_population)
    iterations_label = np.zeros(len(values_history))
    for i in range(len(iterations_label)):
        iterations_label[i] = i
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Number of iterations", fontsize=18)
    ax.set_ylabel("Values", rotation=90, fontsize=18)

    for ye, xe in zip(values_history, iterations_label):
        ax.scatter([xe] * len(ye), ye, c='blue', marker=',')
    plt.title(plots_title, fontweight='bold', fontsize=26)
    plt.savefig(plots_title)
    plt.show()
    return f"Best value: {best_value}"


if __name__ == "__main__":
    print("Probability of mutation (0.0-1.0): ")
    mutation_prob = float(input())
    print("Probabilty of crossover (0.0-1.0): ")
    crossover_prob = float(input())
    print("Size of the starting population (integer): ")
    pop_size = int(input())
    print("Time budget - how many evaluations (integer): ")
    time_bud = int(input())
    test_solver = gen_sol.GeneticSolver(mutation_p=mutation_prob, crossover_p=crossover_prob, time_budget=time_bud)
    print("Please enter the title of the plot:")
    title = str(input())

    print("Creating plot...")
    print(found_optimum_in_time_plot(test_solver, pop_size, title))
