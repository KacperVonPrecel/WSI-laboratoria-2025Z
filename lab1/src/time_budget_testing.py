import genetic_solver as gen_sol
from calc_target import calc_target
import numpy as np


def time_testing(solver: gen_sol.GeneticSolver, pop_size: int, file_name: str, values_to_test: np.ndarray[int]) -> str:
    test_population = gen_sol.create_first_population(pop_size)
    with open(file_name, "w") as file_handler:
        for parameter_value in values_to_test:
            solver.set_time_budget(parameter_value)
            for _ in range(20):
                _, best_value, _ = solver.solve(calc_target, test_population)
                file_handler.write(f"{str(best_value)} ")
        for parameter_value in values_to_test:
            file_handler.write(f'{str(parameter_value)} ')
    return "Done"


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

    print("Input the collection of time budgets you want to test:")
    parameter_values_str = str(input())
    parameter_values = [int(value.strip()) for value in parameter_values_str.split()]
    print("Input the name of the file to save to:")
    file_name = str(input())

    print("Searching...")
    print(time_testing(test_solver, pop_size, file_name, parameter_values))
