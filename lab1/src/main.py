import genetic_solver as gen_sol
from calc_target import calc_target

if __name__ == '__main__':
    test_solver = gen_sol.GeneticSolver(mutation_p=0.05, crossover_p=0.7, time_budget=100000)
    test_population = gen_sol.create_first_population(350)
    best_specimen, best_value = test_solver.solve(calc_target, test_population)
    print(f"Best specimen: {best_specimen} \n Best value: {best_value} \n")
