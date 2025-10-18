import genetic_solver as gen_sol
from calc_target import calc_target
import numpy as np


def checking_loop(solver: gen_sol.GeneticSolver, pop_size: int, file_name: str) -> str:
    test_population = gen_sol.create_first_population(pop_size)
    with open(file_name, "w") as file_handler:
        for _ in range(20):
            _, best_value, _ = solver.solve(calc_target, test_population)
            print(f"Best value: {best_value}")
            file_handler.write(f"{str(best_value)} ")
    return "Done"


def calc_median_and_deviation(file: str) -> tuple[float, float]:
    with open(file, "r") as fh:
        for line in fh:
            values_table = line.strip().split()
        values_array = np.array(values_table, dtype=np.float64)
    median = np.mean(values_array)
    deviation = np.std(values_array, dtype=np.float64)
    return median, deviation


if __name__ == '__main__':
    print("Probability of mutation (0.0-1.0): ")
    mutation_prob = float(input())
    print("Probabilty of crossover (0.0-1.0): ")
    crossover_prob = float(input())
    print("Size of the starting population (integer): ")
    pop_size = int(input())
    print("Time budget - how many evaluations (integer): ")
    time_bud = int(input())
    test_solver = gen_sol.GeneticSolver(mutation_p=mutation_prob, crossover_p=crossover_prob, time_budget=time_bud)

    print("Process with the loop? (Y/N): ")
    decision = str(input())
    if decision == "Y":
        print("Input the name of the file to save to:")
        file_name = str(input())
        print("Searching...")
        print(checking_loop(test_solver, pop_size, file_name))
        print(test_solver.get_parameters())
        print(calc_median_and_deviation(file_name))
    else:
        test_population = gen_sol.create_first_population(pop_size)
        best_specimen, best_value, values_history = test_solver.solve(calc_target, test_population)
        print(f"Best specimen: {best_specimen} \n Best value: {best_value} \n Ranom record: {values_history[2]}")
        print("Oh. Okay.")
