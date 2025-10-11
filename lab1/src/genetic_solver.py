from solver import Solver
from calc_target import calc_target, calc_path
import numpy as np
from typing import Any, Callable, Dict, Tuple


class GeneticSolver(Solver):
    """A genetic solver with mutation, proportional reproduction,
    one-point crossover and generation succession"""

    def __init__(self, n: int = 50, mutation_p: float = 0.05, crossover_p: float = 0.7):
        super().__init__()
        # Population size
        self.pop_size = n
        # Mutation probability
        self.mutation_p = mutation_p
        # Crossover probability
        self.crossover_p = crossover_p
        # Current population
        self.population: np.array[int] = np.zeros(self.pop_size)

    def get_parameters(self) -> Dict[str, Any]:
        param_dict = dict(
            {"population size": self.pop_size},
            {"mutation probability": self.mutation_p},
            {"crossover probabiity": self.crossover_p},
        )
        return param_dict
