import genetic_solver as gen_sol
from calc_target import calc_target

if __name__ == '__main__':
    test_solver = gen_sol.GeneticSolver()
    test_population = gen_sol.create_first_population(20)
    test_value = test_solver.evaluate(calc_target, test_population)
    print(test_value)
    print("\n")
    print(test_solver.find_best(test_value, test_population))
    print("\n")
    test_selection = test_solver.selection(test_value, test_population)
    test_value = test_solver.evaluate(calc_target, test_selection)
    print(test_solver.find_best(test_value, test_selection))
    print("\n")
