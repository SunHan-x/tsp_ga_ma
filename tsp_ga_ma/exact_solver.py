"""
Exact TSP solver using backtracking + branch-and-bound.

This is used as a reference solution for small graphs (e.g. the 12-vertex graph
G_12_66 from the paper).
"""

from typing import List, Tuple
import math
from .utils import tour_length


def exact_tsp_backtracking(dist_matrix) -> Tuple[List[int], int, int]:
    """
    Solve TSP exactly using backtracking + branch-and-bound.

    We fix the starting vertex as 0 and explore all permutations of the
    remaining vertices, pruning branches when the optimistic lower bound
    exceeds the current best cost.

    Returns:
        best_tour: list of vertex indices representing the best cycle
        best_cost: optimal cost
        generated: number of partial/complete solutions visited
    """
    n = len(dist_matrix)
    best_tour = None
    best_cost = math.inf
    visited = [False] * n
    visited[0] = True
    current_path: List[int] = [0]
    generated = 0

    # Pre-compute a minimal outgoing edge cost for each vertex
    min_out = []
    for i in range(n):
        row = dist_matrix[i]
        min_val = min(row[j] for j in range(n) if j != i)
        min_out.append(min_val)

    def backtrack(last: int, depth: int, current_cost: int):
        nonlocal best_tour, best_cost, generated

        generated += 1

        # If we have visited all vertices, close the cycle
        if depth == n:
            total_cost = current_cost + dist_matrix[last][0]
            if total_cost < best_cost:
                best_cost = total_cost
                best_tour = current_path.copy()
            return

        # Compute a simple optimistic lower bound:
        # current_cost + sum of minimal outgoing edges for unvisited vertices
        remaining_lower_bound = 0
        for v in range(n):
            if not visited[v]:
                remaining_lower_bound += min_out[v]

        if current_cost + remaining_lower_bound >= best_cost:
            # Prune this branch
            return

        # Try all unvisited vertices
        for v in range(1, n):
            if not visited[v]:
                visited[v] = True
                current_path.append(v)
                backtrack(v, depth + 1, current_cost + dist_matrix[last][v])
                current_path.pop()
                visited[v] = False

    backtrack(0, 1, 0)
    return best_tour, best_cost, generated
