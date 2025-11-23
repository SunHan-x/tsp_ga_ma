"""
Local search method used inside the Memetic Algorithm (MA).

The idea follows the paper:
- Split a tour into equal-length groups (e.g., 3 groups for 12 vertices)
- For each group, try all cyclic shifts
- Evaluate the full tour for each shifted group
- Keep the best shift that improves (or keeps) the current best cost
"""

from typing import List
from .utils import tour_length


def local_search_rotate_groups(tour: List[int],
                               dist_matrix,
                               groups: int = 3) -> List[int]:
    """
    Perform local search by rotating each group in the tour.

    If n is divisible by `groups`, we:
      - split the tour into `groups` segments of equal length
      - for each segment, try all cyclic rotations and keep the one that
        yields the best total tour length (global evaluation)

    If n is not divisible by `groups`, the function returns the original tour.

    This procedure never worsens the current solution: it either returns a
    strictly better tour or the same one.
    """
    n = len(tour)
    if n % groups != 0:
        # For simplicity, only handle equal-sized groups
        return tour

    group_size = n // groups
    best_tour = tour.copy()
    best_cost = tour_length(best_tour, dist_matrix)

    # Process each group in sequence
    for g in range(groups):
        start = g * group_size
        end = start + group_size

        group = best_tour[start:end]
        current_group_best = group.copy()
        current_group_best_cost = best_cost

        # Try all cyclic shifts of this group
        for shift in range(1, group_size):
            rotated = group[shift:] + group[:shift]
            candidate = best_tour.copy()
            candidate[start:end] = rotated
            cost = tour_length(candidate, dist_matrix)
            if cost < current_group_best_cost:
                current_group_best_cost = cost
                current_group_best = rotated

        # Fix the best rotation for this group
        best_tour[start:end] = current_group_best
        best_cost = current_group_best_cost

    return best_tour
