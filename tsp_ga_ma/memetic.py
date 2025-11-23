"""
Memetic Algorithm (MA) for TSP.

This is essentially:
    Genetic Algorithm + local search.

We use the same GA structure as in ga.py, but after crossover we apply
local search on (some or all) offspring to refine them.
"""

from typing import List, Dict
import random
import time
from .utils import tour_length
from .ga import crossover_one_point, mutate_swap
from .local_search import local_search_rotate_groups


def memetic_tsp(dist_matrix,
                population_size: int = 8,
                iterations: int = 100,
                mutation_rate: float = 0.1,
                local_search_prob: float = 1.0,
                groups_for_ls: int = 3,
                seed: int | None = None) -> Dict:
    """
    Memetic Algorithm for TSP.

    Args:
        dist_matrix: distance matrix of the graph
        population_size: number of individuals in the population
        iterations: number of generations
        mutation_rate: probability of mutating each offspring
        local_search_prob: probability of applying local search to each offspring
        groups_for_ls: number of groups for the local search procedure
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

    # Initialize population
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
        population.sort(key=fitness)
        current_best = population[0]
        current_cost = fitness(current_best)

        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = current_best.copy()

        history.append(best_cost)

        parents = population[: population_size // 2]

        new_children: List[List[int]] = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % len(parents)]
            point = n // 2
            c1, c2 = crossover_one_point(p1, p2, point)
            crossover_solutions += 2
            new_children.extend([c1, c2])

        # Local search (Step 7.1 in the paper)
        for i in range(len(new_children)):
            if random.random() < local_search_prob:
                new_children[i] = local_search_rotate_groups(
                    new_children[i],
                    dist_matrix,
                    groups=groups_for_ls
                )

        # Mutation
        for child in new_children:
            if random.random() < mutation_rate:
                mutate_swap(child)
                mutation_solutions += 1

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
