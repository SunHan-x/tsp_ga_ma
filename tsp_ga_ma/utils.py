"""
Utility functions for TSP tours.
"""

from typing import List
from .data import vertex_labels_g12


def tour_length(tour: List[int], dist_matrix) -> int:
    """
    Compute the length of a Hamiltonian cycle represented by a list of vertex indices.
    The tour is assumed to be a cycle: last vertex connects back to the first.
    """
    n = len(tour)
    total = 0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        total += dist_matrix[a][b]
    return total


def format_tour(tour: List[int], labels=None) -> str:
    """
    Format a tour as (a)–(b)–...–(a), using given labels or default G_12_66 labels.
    """
    if labels is None:
        labels = vertex_labels_g12
    return "–".join(f"({labels[i]})" for i in tour) + f"–({labels[tour[0]]})"
