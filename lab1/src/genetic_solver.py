from solver import Solver
from calc_target import calc_target, calc_path
import numpy as np
from typing import Any, Callable, Dict, Tuple


class GeneticSolver(Solver):
    """A genetic solver with mutation, proportional reproduction,
    one-point crossover and generation succession"""

    def __init__(self, n: int = 50, mutation_p: float = 0.05, crossover_p: float = 0.7, time_budget: int = 50000):
        super().__init__()
        # Population size
        self.pop_size = n
        # Mutation probability
        self.mutation_p = mutation_p
        # Crossover probability
        self.crossover_p = crossover_p
        # Current population
        self.population: np.array[int] = np.zeros((self.pop_size, 200, 2))
        # FES for this solver
        self.time_budget = time_budget

    def get_parameters(self) -> Dict[str, Any]:
        param_dict = {
            "population size": self.pop_size,
            "mutation probability": self.mutation_p,
            "crossover probabiity": self.crossover_p,
            "FES": self.time_budget
        }
        return param_dict

    def create_first_population(self) -> None:
        for specimen in self.population:
            t = 0
            while t < len(specimen):
                specimen[..., t, 0] = np.random.randint(2)
                specimen[..., t, 1] = np.random.randint(2)
                t += 1

    def evaluate(self, func: Callable[[np.ndarray], float]) -> np.ndarray[float]:
        values = np.zeros(self.pop_size)
        for i in self.population:
            values[i] = func(i)
        return values

    def find_best(self, values: np.ndarray[float]) -> Tuple[np.ndarray, float]:
        best = np.argmax(values)
        return self.population[best], values[best]

    def selection(self, values: np.ndarray[float]):
        eps = 1e-9
        shifted = values - np.min(values) + eps
        probabilities = shifted / np.sum(shifted)
        # Population after selection
        self.population = np.random.choice(len(values), size=self.pop_size, p=probabilities)

    def solve(self, func: Callable[[np.ndarray], int]) -> Tuple[np.ndarray, float]:
        return super().solve()
