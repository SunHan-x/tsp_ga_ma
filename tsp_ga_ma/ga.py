"""
Genetic Algorithm (GA) for solving the TSP.

This follows the high-level idea in the paper:
- representation: permutation of vertices
- fitness: total tour length (to be minimized)
- selection: truncation (keep best half)
- crossover: one-point crossover with repair to ensure valid permutations
- mutation: swap mutation
"""

from typing import List, Dict
import random
import time
from .utils import tour_length


def crossover_one_point(parent1: List[int],
                        parent2: List[int],
                        point: int) -> (List[int], List[int]):
    """
    One-point crossover with simple repair: generate two valid children.

    child1 = parent1[:point] + remaining genes from parent2 (in the same order)
    child2 = parent2[:point] + remaining genes from parent1 (in the same order)

    This ensures each city appears exactly once in the permutation.
    """
    n = len(parent1)

    # Child 1
    child1_first = parent1[:point]
    used1 = set(child1_first)
    child1 = child1_first + [g for g in parent2 if g not in used1]

    # Child 2
    child2_first = parent2[:point]
    used2 = set(child2_first)
    child2 = child2_first + [g for g in parent1 if g not in used2]

    return child1, child2


def mutate_swap(tour: List[int]) -> None:
    """
    Simple swap mutation: randomly exchange two positions in the tour.
    """
    n = len(tour)
    i, j = random.sample(range(n), 2)
    tour[i], tour[j] = tour[j], tour[i]


def genetic_tsp(dist_matrix,
                population_size: int = 8,
                iterations: int = 100,
                mutation_rate: float = 0.1,
                seed: int | None = None) -> Dict:
    """
    Genetic Algorithm for TSP.

    Args:
        dist_matrix: distance matrix of the graph
        population_size: number of individuals in the population
        iterations: number of generations
        mutation_rate: probability of mutating each offspring
        seed: optional random seed

    Returns:
        A dict containing:
            best_tour, best_cost, history,
            crossover_solutions, mutation_solutions,
            total_solutions, time_ms
    """
    if seed is not None:
        random.seed(seed)

    n = len(dist_matrix)

    # ---- Step 1: initialize population ----
    population: List[List[int]] = []
    for _ in range(population_size):
        tour = list(range(n))
        random.shuffle(tour)
        population.append(tour)

    def fitness(t: List[int]) -> int:
        return tour_length(t, dist_matrix)

    start_time = time.time()

    best_tour = None
    best_cost = float("inf")

    crossover_solutions = 0
    mutation_solutions = 0

    history: List[int] = []

    for _ in range(iterations):
        # ---- Step 2: evaluate and sort ----
        population.sort(key=fitness)
        current_best = population[0]
        current_cost = fitness(current_best)

        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = current_best.copy()

        history.append(best_cost)

        # ---- Step 3: selection (truncate best half) ----
        parents = population[: population_size // 2]

        # ---- Step 4: crossover ----
        new_children: List[List[int]] = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % len(parents)]
            point = n // 2  # middle crossover point, as in the paper example
            c1, c2 = crossover_one_point(p1, p2, point)
            new_children.extend([c1, c2])
            crossover_solutions += 2

        # ---- Step 6: mutation ----
        for child in new_children:
            if random.random() < mutation_rate:
                mutate_swap(child)
                mutation_solutions += 1

        # ---- Step 7: form new population ----
        population = parents + new_children
        population = population[:population_size]

    end_time = time.time()
    time_ms = int((end_time - start_time) * 1000)
    total_solutions = crossover_solutions + mutation_solutions

    return {
        "best_tour": best_tour,
        "best_cost": best_cost,
        "history": history,
        "crossover_solutions": crossover_solutions,
        "mutation_solutions": mutation_solutions,
        "total_solutions": total_solutions,
        "time_ms": time_ms,
    }
