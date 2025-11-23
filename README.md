# TSP with Genetic Algorithm and Memetic Algorithm

This project is a small Python implementation inspired by the paper:

> Kralev, T., & Kraleva, R. (2024).  
> *Combining Genetic Algorithm with Local Search Method in Solving Optimization Problems*.  
> Electronics 13(20), 4126.

The code compares:

- An **exact solver** (backtracking + branch-and-bound) on small instances
- A basic **Genetic Algorithm (GA)** for TSP
- A **Memetic Algorithm (MA)** = GA + local search

The main example reproduces the experiments on the 12-vertex graph `G_12_66`
from the paper.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python examples/example_g12_ga_vs_ma.py
