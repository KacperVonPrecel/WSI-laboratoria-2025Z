import genetic_solver as gen_sol
from calc_target import calc_target
import numpy as np


def population_testing(solver: gen_sol.GeneticSolver, pop_size_values: np.ndarray[int], file_name: str) -> str:
    np.random.seed(1234)
    with open(file_name, "w") as file_handler:
        for pop_size in pop_size_values:
            test_population = gen_sol.create_first_population(pop_size)
            for _ in range(20):
                _, best_value, _ = solver.solve(calc_target, test_population)
                file_handler.write(f"{str(best_value)} ")
            file_handler.write("\n")
        for pop_size in pop_size_values:
            file_handler.write(f'{str(pop_size)} ')
    return "Done"


if __name__ == '__main__':
    print("Probability of mutation (0.0-1.0): ")
    mutation_prob = float(input())
    print("Probabilty of crossover (0.0-1.0): ")
    crossover_prob = float(input())
    print("Time budget - how many evaluations (integer): ")
    time_bud = int(input())
    test_solver = gen_sol.GeneticSolver(mutation_p=mutation_prob, crossover_p=crossover_prob, time_budget=time_bud)

    print("Input the collection of population sizes you want to test:")
    parameter_values_str = str(input())
    parameter_values = [int(value.strip()) for value in parameter_values_str.split()]
    print("Input the name of the file to save to:")
    file_name = str(input())

    print("Searching...")
    print(population_testing(test_solver, parameter_values, file_name))
