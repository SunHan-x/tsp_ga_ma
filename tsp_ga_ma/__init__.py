"""
TSP solved with Genetic Algorithm and Memetic Algorithm.

This package provides:
- An exact backtracking + branch-and-bound solver for small TSP instances
- A Genetic Algorithm (GA) implementation
- A Memetic Algorithm (MA) implementation combining GA with local search
"""

from .data import vertex_labels_g12, dist_matrix_g12
from .utils import tour_length, format_tour
from .exact_solver import exact_tsp_backtracking
from .ga import genetic_tsp
from .memetic import memetic_tsp
