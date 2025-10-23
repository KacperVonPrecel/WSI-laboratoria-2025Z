from solver import Solver
# from calc_target import calc_target, calc_path
import numpy as np
from typing import Any, Callable, Dict, Tuple

INPUTS_NUM = 400


class GeneticSolver(Solver):
    """A genetic solver with mutation, proportional reproduction,
    one-point crossover and generation succession"""

    def __init__(self, mutation_p: float = 0.05, crossover_p: float = 0.85, time_budget: int = 50000):
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

    def set_mutation_p(self, new_prob: float):
        self.mutation_p = new_prob

    def set_crossover_p(self, new_prob: float):
        self.crossover_p = new_prob

    def set_time_budget(self, new_FES: int):
        self.time_budget = new_FES

    def evaluate(self, func: Callable[[np.ndarray], float], population: np.ndarray[int]) -> np.ndarray[float]:
        values = np.zeros(len(population))
        pos = 0
        for i in population:
            #           Soft limitations:
            # ones_num = 0
            # penalty = 0
            # for bit in i:
            #     if bit == 1:
            #         ones_num += 1
            # if ones_num > 300:
            #     penalty = -(ones_num - 300)**2
            # values[pos] = func(i) + penalty
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
                while not check_slices(first_parent, second_parent, crossover_point):
                    crossover_point = np.random.randint(1, 399)
                tmp_slice = first_parent[crossover_point:].copy()
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
            ones_count = int(np.sum(offspring))
            for i in range(len(offspring)):
                if np.random.random() < self.mutation_p:
                    if offspring[i] == 1:
                        offspring[i] = 0
                        ones_count -= 1
                    elif ones_count < 300:
                        offspring[i] = 1
                        ones_count += 1

        return new_generation

    def solve(self, func: Callable[[np.ndarray], int], first_population: np.ndarray[int]) -> Tuple[np.ndarray, float, np.ndarray[float]]:
        time = 0
        time_max = self.time_budget // len(first_population)
        values_history = np.zeros(((1 + time_max), len(first_population)))
        values = self.evaluate(func, first_population)
        values_history[0] = values
        best_sepcimen, best_value = self.find_best(values, first_population)
        while time < time_max:
            selected_population = self.selection(values, first_population)
            next_generation = self.crossover_and_mutation(selected_population)
            values = self.evaluate(func, next_generation)
            values_history[time + 1] = values
            current_specimen, current_value = self.find_best(values, next_generation)
            if current_value > best_value:
                best_sepcimen = current_specimen
                best_value = current_value
            first_population = next_generation
            time += 1

        return best_sepcimen, best_value, values_history


def create_first_population(pop_size: int = 50) -> np.ndarray[int]:
    first_population = np.zeros((pop_size, INPUTS_NUM))
    for specimen in first_population:
        ones_count = 0
        t = 0
        while t < len(specimen):
            if ones_count == 300:
                specimen[t] = 0
            else:
                specimen[t] = np.random.randint(2)
                if specimen[t] == 1:
                    ones_count += 1
            t += 1
    return first_population


def check_slices(first_parent: np.ndarray[int], second_parent: np.ndarray[int], crossover_point: int) -> bool:
    tmp_slice = first_parent[crossover_point:].copy()
    first_parent[crossover_point:] = second_parent[crossover_point:]
    second_parent[crossover_point:] = tmp_slice

    ones_count_first_parent = int(np.sum(first_parent))
    ones_count_second_parent = int(np.sum(second_parent))

    return ones_count_first_parent <= 300 and ones_count_second_parent <= 300
