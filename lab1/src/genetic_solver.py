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

    def crossover_and_mutation(self, selected_pops: np.ndarray[int]) -> np.ndarray[int]:
        # CROSSOVER
        # matched parents - an array in which two consecutive specimen are in pair
        matched_parents_pos = np.random.choice(len(selected_pops), size=len(selected_pops), replace=False)
        matched_parents = np.zeros((len(selected_pops), INPUTS_NUM))
        pos = 0
        for parent_pos in matched_parents_pos:
            matched_parents[pos] = selected_pops[parent_pos]
            pos += 1
        pos = 0
        tmp_list_of_children = []
        pos_limit_sub = 1
        if len(matched_parents) % 2 == 1:
            pos_limit_sub = 2

        while pos < len(matched_parents) - pos_limit_sub:
            first_parent = matched_parents[pos]
            second_parent = matched_parents[pos + 1]
            if np.random.choice([0, 1], p=[1 - self.crossover_p, self.crossover_p]) == 1:
                crossover_point = np.random.randint(1, 399)
                tmp_slice = first_parent[crossover_point:]
                first_parent[crossover_point:] = second_parent[crossover_point:]
                second_parent[crossover_point:] = tmp_slice
            tmp_list_of_children.append(first_parent)
            tmp_list_of_children.append(second_parent)
            pos += 2

        if pos_limit_sub == 3:
            tmp_list_of_children.append(matched_parents[-1])

        # MUTATION
        new_generation = np.array(tmp_list_of_children)
        for offspring in new_generation:
            for cell in offspring:
                if np.random.choice([0, 1], p=[1 - self.mutation_p, self.mutation_p]) == 1:
                    cell = 1 if cell == 0 else 0

        return new_generation

    def solve(self, func: Callable[[np.ndarray], int], first_population: np.ndarray[int]) -> Tuple[np.ndarray, float]:
        time = 0
        values = self.evaluate(func, first_population)
        best_sepcimen, best_value = self.find_best(values, first_population)
        time_max = self.time_budget // len(first_population)
        while time < time_max:
            selected_population = self.selection(values, first_population)
            next_generation = self.crossover_and_mutation(selected_population)
            values = self.evaluate(func, next_generation)
            current_specimen, current_value = self.find_best(values, next_generation)
            if current_value > best_value:
                best_sepcimen = current_specimen
                best_value = current_value
            first_population = next_generation
            time += 1

        return best_sepcimen, best_value


def create_first_population(pop_size: int = 50) -> np.ndarray[int]:
    first_population = np.zeros((pop_size, INPUTS_NUM))
    for specimen in first_population:
        t = 0
        while t < len(specimen):
            specimen[t] = np.random.randint(2)
            t += 1
    return first_population
