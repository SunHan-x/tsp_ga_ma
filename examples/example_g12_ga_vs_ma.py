"""
Example script: compare Exact, GA, and MA on the 12-vertex graph G_12_66.

Run:

    python examples/example_g12_ga_vs_ma.py
"""

import sys
import os
import argparse
import time

# Add the parent directory to sys.path to allow importing tsp_ga_ma
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tsp_ga_ma import (
    dist_matrix_g12,
    vertex_labels_g12,
    tour_length,
    format_tour,
    exact_tsp_backtracking,
    genetic_tsp,
    memetic_tsp,
)

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Compare Exact, GA, and MA on G_12_66")
    parser.add_argument("--pop_size", type=int, default=100, help="Population size")
    parser.add_argument("--iterations", type=int, default=500, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=999, help="Random seed")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate")
    args = parser.parse_args()

    print(f"Parameters: pop_size={args.pop_size}, iterations={args.iterations}, seed={args.seed}, mutation_rate={args.mutation_rate}")
    print()

    print("=== Exact TSP on G_12_66 ===")
    start_time = time.time()
    best_tour, best_cost, generated = exact_tsp_backtracking(dist_matrix_g12)
    end_time = time.time()
    exact_time_ms = int((end_time - start_time) * 1000)
    print("Best cost:", best_cost)
    print("Best tour:", format_tour(best_tour, vertex_labels_g12))
    print("Generated solutions:", generated)
    print("Exact time (ms):", exact_time_ms)
    print()

    print("=== Genetic Algorithm (GA) ===")
    ga_res = genetic_tsp(
        dist_matrix_g12,
        population_size=args.pop_size,
        iterations=args.iterations,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
    )
    print("GA best cost:", ga_res["best_cost"])
    print("GA best tour:", format_tour(ga_res["best_tour"], vertex_labels_g12))
    print("GA crossover solutions:", ga_res["crossover_solutions"])
    print("GA mutation solutions:", ga_res["mutation_solutions"])
    print("GA total solutions:", ga_res["total_solutions"])
    print("GA time (ms):", ga_res["time_ms"])
    print()

    print("=== Memetic Algorithm (MA) - Rotate Groups ===")
    ma_res = memetic_tsp(
        dist_matrix_g12,
        population_size=args.pop_size,
        iterations=args.iterations,
        mutation_rate=args.mutation_rate,
        local_search_prob=1.0,
        groups_for_ls=3,
        ls_method="rotate",
        seed=args.seed,
    )
    print("MA (Rotate) best cost:", ma_res["best_cost"])
    print("MA (Rotate) best tour:", format_tour(ma_res["best_tour"], vertex_labels_g12))
    print("MA (Rotate) crossover solutions:", ma_res["crossover_solutions"])
    print("MA (Rotate) mutation solutions:", ma_res["mutation_solutions"])
    print("MA (Rotate) total solutions:", ma_res["total_solutions"])
    print("MA (Rotate) time (ms):", ma_res["time_ms"])
    print()

    # print("=== Memetic Algorithm (MA) - Rotate Groups (Iterative) ===")
    # ma_iter_res = memetic_tsp(
    #     dist_matrix_g12,
    #     population_size=args.pop_size,
    #     iterations=args.iterations,
    #     mutation_rate=args.mutation_rate,
    #     local_search_prob=1.0,
    #     groups_for_ls=3,
    #     ls_method="rotate_iterative",
    #     seed=args.seed,
    # )
    # print("MA (Rotate Iter) best cost:", ma_iter_res["best_cost"])
    # print("MA (Rotate Iter) best tour:", format_tour(ma_iter_res["best_tour"], vertex_labels_g12))
    # print("MA (Rotate Iter) crossover solutions:", ma_iter_res["crossover_solutions"])
    # print("MA (Rotate Iter) mutation solutions:", ma_iter_res["mutation_solutions"])
    # print("MA (Rotate Iter) total solutions:", ma_iter_res["total_solutions"])
    # print("MA (Rotate Iter) time (ms):", ma_iter_res["time_ms"])
    # print()

    print("=== Memetic Algorithm (MA) - Rotate Groups (Dynamic) ===")
    ma_dyn_res = memetic_tsp(
        dist_matrix_g12,
        population_size=args.pop_size,
        iterations=args.iterations,
        mutation_rate=args.mutation_rate,
        local_search_prob=1.0,
        groups_for_ls=3, # Ignored for dynamic
        ls_method="rotate_dynamic",
        seed=args.seed,
    )
    print("MA (Rotate Dyn) best cost:", ma_dyn_res["best_cost"])
    print("MA (Rotate Dyn) best tour:", format_tour(ma_dyn_res["best_tour"], vertex_labels_g12))
    print("MA (Rotate Dyn) crossover solutions:", ma_dyn_res["crossover_solutions"])
    print("MA (Rotate Dyn) mutation solutions:", ma_dyn_res["mutation_solutions"])
    print("MA (Rotate Dyn) total solutions:", ma_dyn_res["total_solutions"])
    print("MA (Rotate Dyn) time (ms):", ma_dyn_res["time_ms"])
    print()

    print("=== Memetic Algorithm (MA) - 2-opt ===")
    ma_2opt_res = memetic_tsp(
        dist_matrix_g12,
        population_size=args.pop_size,
        iterations=args.iterations,
        mutation_rate=args.mutation_rate,
        local_search_prob=1.0,
        groups_for_ls=3,
        ls_method="2opt",
        seed=args.seed,
    )
    print("MA (2-opt) best cost:", ma_2opt_res["best_cost"])
    print("MA (2-opt) best tour:", format_tour(ma_2opt_res["best_tour"], vertex_labels_g12))
    print("MA (2-opt) crossover solutions:", ma_2opt_res["crossover_solutions"])
    print("MA (2-opt) mutation solutions:", ma_2opt_res["mutation_solutions"])
    print("MA (2-opt) total solutions:", ma_2opt_res["total_solutions"])
    print("MA (2-opt) time (ms):", ma_2opt_res["time_ms"])
    print()

    # Plot convergence curves (best cost over iterations)
    plt.figure()
    plt.axhline(y=best_cost, color='r', linestyle='--', label='Exact')
    plt.plot(ga_res["history"], label="GA")
    plt.plot(ma_res["history"], label="MA (Rotate)")
    # plt.plot(ma_iter_res["history"], label="MA (Rotate Iter)")
    plt.plot(ma_dyn_res["history"], label="MA (Rotate Dyn)")
    plt.plot(ma_2opt_res["history"], label="MA (2-opt)")
    plt.xlabel("Iteration")
    plt.ylabel("Best cost so far")
    plt.title(f"GA vs MA Variants\n(Pop={args.pop_size}, Iter={args.iterations}, Seed={args.seed})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Ensure result directory exists
    os.makedirs("result", exist_ok=True)
    
    # plt.show() # Commented out to avoid blocking batch execution
    plot_filename = f"result/convergence_p{args.pop_size}_i{args.iterations}_s{args.seed}.png"
    plt.savefig(plot_filename)
    print(f"Saved plot to {plot_filename}")


if __name__ == "__main__":
    main()
