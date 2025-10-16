import genetic_solver as gen_sol
from calc_target import calc_target

if __name__ == '__main__':
    print("Probability of mutation (0.0-1.0): ")
    mutation_prob = float(input())
    print("Probabilty of crossover (0.0-1.0): ")
    crossover_prob = float(input())
    print("Size of the starting population (integer): ")
    pop_size = int(input())
    print("Time budget - how many evaluations (integer): ")
    time_bud = int(input())
    print("Searching...")

    test_solver = gen_sol.GeneticSolver(mutation_p=mutation_prob, crossover_p=crossover_prob, time_budget=time_bud)
    test_population = gen_sol.create_first_population(pop_size)
    best_specimen, best_value = test_solver.solve(calc_target, test_population)
    print(f"Best specimen: {best_specimen} \n Best value: {best_value} \n")
