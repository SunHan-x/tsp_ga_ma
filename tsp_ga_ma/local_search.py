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


def local_search_2opt(tour: List[int], dist_matrix) -> List[int]:
    """
    Perform 2-opt local search.
    Iteratively reverse segments of the tour to reduce length.
    """
    n = len(tour)
    best_tour = tour.copy()
    improved = True

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Skip if j is the last node and i is the first (edge (n-1, 0) and (0, 1) are adjacent)
                if i == 0 and j == n - 1:
                    continue

                u, v = best_tour[i], best_tour[i + 1]
                x, y = best_tour[j], best_tour[(j + 1) % n]

                delta = -dist_matrix[u][v] - dist_matrix[x][y] + dist_matrix[u][x] + dist_matrix[v][y]

                if delta < 0:
                    best_tour[i + 1 : j + 1] = reversed(best_tour[i + 1 : j + 1])
                    improved = True
                    # Restart search after improvement (First Improvement)
                    # break
            # if improved: break

    return best_tour


def local_search_rotate_groups_iterative(tour: List[int],
                                         dist_matrix,
                                         groups: int = 3) -> List[int]:
    """
    Perform local search by rotating each group in the tour iteratively until no improvement.
    """
    current_tour = tour.copy()
    current_cost = tour_length(current_tour, dist_matrix)
    
    while True:
        # Perform one pass of group rotations
        new_tour = local_search_rotate_groups(current_tour, dist_matrix, groups)
        new_cost = tour_length(new_tour, dist_matrix)
        
        if new_cost < current_cost:
            current_tour = new_tour
            current_cost = new_cost
        else:
            # No improvement in this full pass
            break
            
    return current_tour


def local_search_rotate_groups_dynamic(tour: List[int],
                                       dist_matrix,
                                       group_sizes: List[int] = [3, 4]) -> List[int]:
    """
    Perform local search by rotating groups with dynamic sizes.
    It iterates through the list of group counts provided in `group_sizes`.
    For each group count, it performs the rotation search.
    This process repeats until no improvement is found across all group sizes.
    """
    current_tour = tour.copy()
    current_cost = tour_length(current_tour, dist_matrix)
    
    while True:
        improved_in_cycle = False
        
        for groups in group_sizes:
            # Perform one pass of group rotations with current group count
            new_tour = local_search_rotate_groups(current_tour, dist_matrix, groups)
            new_cost = tour_length(new_tour, dist_matrix)
            
            if new_cost < current_cost:
                current_tour = new_tour
                current_cost = new_cost
                improved_in_cycle = True
        
        if not improved_in_cycle:
            break
            
    return current_tour
