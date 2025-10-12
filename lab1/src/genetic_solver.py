from solver import Solver
# from calc_target import calc_target, calc_path
import numpy as np
from typing import Any, Callable, Dict, Tuple

INPUTS_NUM = 400


class GeneticSolver(Solver):
    """A genetic solver with mutation, proportional reproduction,
    one-point crossover and generation succession"""

    def __init__(self, mutation_p: float = 0.05, crossover_p: float = 0.7, time_budget: int = 50000):
        super().__init__()
        # Mutation probability
        self.mutation_p = mutation_p
        # Crossover probability
        self.crossover_p = crossover_p
        # FES for this solver
        self.time_budget = time_budget

    def get_parameters(self) -> Dict[str, Any]:
        param_dict = {
            "mutation probability": self.mutation_p,
            "crossover probabiity": self.crossover_p,
            "FES": self.time_budget
        }
        return param_dict

    def evaluate(self, func: Callable[[np.ndarray], float], population: np.ndarray[int]) -> np.ndarray[float]:
        values = np.zeros(len(population))
        pos = 0
        for i in population:
            values[pos] = func(i)
            pos += 1
        return values

    def find_best(self, values: np.ndarray[float], population: np.ndarray[int]) -> Tuple[np.ndarray, float]:
        best = np.argmax(values)
        return population[best], values[best]

    def selection(self, values: np.ndarray[float], population: np.ndarray[int]) -> np.ndarray[int]:
        eps = 1e-9
        selected_candidates = np.zeros((len(population), INPUTS_NUM))
        pos = 0
        shifted = values - np.min(values) + eps
        probabilities = shifted / np.sum(shifted)
        # Population after selection
        selected_pos = np.random.choice(len(values), size=len(population), p=probabilities)
        for i in selected_pos:
            selected_candidates[pos] = population[i]
            pos += 1
        return selected_candidates

    def solve(self, func: Callable[[np.ndarray], int]) -> Tuple[np.ndarray, float]:
        return super().solve()


def create_first_population(pop_size: int = 50) -> np.ndarray[int]:
    first_population = np.zeros((pop_size, INPUTS_NUM))
    for specimen in first_population:
        t = 0
        while t < len(specimen):
            specimen[t] = np.random.randint(2)
            t += 1
    return first_population
