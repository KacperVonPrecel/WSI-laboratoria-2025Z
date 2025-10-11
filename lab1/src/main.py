import genetic_solver as gen_sol
from calc_target import calc_target

if __name__ == '__main__':
    test_solver = gen_sol.GeneticSolver()
    test_solver.create_first_population()
    print(test_solver.get_parameters())
    print(test_solver.evaluate(calc_target))

# TO DO: Format osobnika jest zły, przeanalizuj calc_path