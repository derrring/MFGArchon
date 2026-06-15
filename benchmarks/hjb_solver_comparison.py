#!/usr/bin/env python3
"""
Performance benchmark comparing HJB solver methods.

Compares Fixed-Point vs Newton solvers on:
- 1D problems (varying grid sizes)
- 2D problems (varying grid sizes)
- Time per iteration
- Convergence speed
- Memory usage
"""

import time

import pandas as pd

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


class QuadraticHamiltonian2D(MFGProblem):
    """2D LQ problem for benchmarking."""

    def __init__(self, N=20, T=1.0, Nt=20, nu=0.01):
        super().__init__(
            spatial_bounds=[(-1, 1), (-1, 1)],
            spatial_discretization=[N, N],
            T=T,
            Nt=Nt,
            sigma=nu,
        )
        self.grid_resolution = N

    def hamiltonian(self, x, m, p, t):
        return 0.5 * np.sum(p**2) + 0.5 * np.sum(x**2)

    def terminal_cost(self, x):
        return 0.5 * np.sum(x**2)

    def initial_density(self, x):
        return np.exp(-5 * np.sum(x**2))

    def running_cost(self, x, m, t):
        return 0.5 * np.sum(x**2)

    def setup_components(self):
        pass


def benchmark_1d_solver(Nx=100, Nt=50, solver_type="newton"):
    """Benchmark 1D HJB solver."""
    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=Nt,
        T=1.0,
    )

    if solver_type == "fixed_point":
        solver = HJBFDMSolver(
            problem,
            solver_type="fixed_point",
            damping_factor=0.8,
            max_newton_iterations=100,
            newton_tolerance=1e-6,
        )
    else:
        solver = HJBFDMSolver(problem, solver_type="newton", max_newton_iterations=30, newton_tolerance=1e-6)

    # Create test inputs
    bounds = problem.geometry.get_bounds()
    Nx_points = problem.geometry.get_grid_shape()[0]
    M = np.ones((Nt + 1, Nx_points)) / Nx_points
    x = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
    U_terminal = 0.5 * x**2
    U_guess = np.zeros((Nt + 1, Nx_points))

    # Benchmark
    start = time.time()
    _U_solution = solver.solve_hjb_system(M, U_terminal, U_guess)
    elapsed = time.time() - start

    # Calculate stats
    grid_points = Nx_points * (Nt + 1)
    time_per_point = elapsed / grid_points * 1e6  # microseconds

    return {
        "dimension": 1,
        "solver_type": solver_type,
        "Nx": Nx,
        "Nt": Nt,
        "total_points": grid_points,
        "total_time": elapsed,
        "time_per_point_us": time_per_point,
        "time_per_timestep": elapsed / Nt,
    }


def benchmark_2d_solver(N=20, Nt=20, solver_type="newton"):
    """Benchmark 2D HJB solver."""
    problem = QuadraticHamiltonian2D(N=N, T=1.0, Nt=Nt)

    if solver_type == "fixed_point":
        solver = HJBFDMSolver(
            problem,
            solver_type="fixed_point",
            damping_factor=0.8,
            max_newton_iterations=100,
            newton_tolerance=1e-6,
        )
    else:
        solver = HJBFDMSolver(problem, solver_type="newton", max_newton_iterations=30, newton_tolerance=1e-6)

    # Create test inputs
    M = np.ones((Nt + 1, N, N)) / (N * N)
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    U_terminal = 0.5 * (X**2 + Y**2)
    U_guess = np.zeros((Nt + 1, N, N))

    # Benchmark
    start = time.time()
    _U_solution = solver.solve_hjb_system(M, U_terminal, U_guess)
    elapsed = time.time() - start

    # Calculate stats
    grid_points = N * N * (Nt + 1)
    time_per_point = elapsed / grid_points * 1e6  # microseconds

    return {
        "dimension": 2,
        "solver_type": solver_type,
        "N": N,
        "Nt": Nt,
        "total_points": grid_points,
        "total_time": elapsed,
        "time_per_point_us": time_per_point,
        "time_per_timestep": elapsed / Nt,
    }


def run_1d_benchmarks():
    """Run comprehensive 1D benchmarks."""
    print("=" * 70)
    print("1D HJB Solver Benchmarks")
    print("=" * 70)

    results = []

    grid_sizes = [50, 100, 200]
    time_steps = [20, 50]
    solver_types = ["fixed_point", "newton"]

    for Nx in grid_sizes:
        for Nt in time_steps:
            for solver_type in solver_types:
                print(f"  Running: 1D, Nx={Nx}, Nt={Nt}, {solver_type}...", end=" ")
                result = benchmark_1d_solver(Nx, Nt, solver_type)
                results.append(result)
                print(f"{result['total_time']:.3f}s")

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    print("\n")

    return df


def run_2d_benchmarks():
    """Run comprehensive 2D benchmarks."""
    print("=" * 70)
    print("2D HJB Solver Benchmarks")
    print("=" * 70)

    results = []

    grid_sizes = [10, 15, 20]
    time_steps = [10, 20]
    solver_types = ["fixed_point", "newton"]

    for N in grid_sizes:
        for Nt in time_steps:
            for solver_type in solver_types:
                print(f"  Running: 2D, N={N}, Nt={Nt}, {solver_type}...", end=" ")
                result = benchmark_2d_solver(N, Nt, solver_type)
                results.append(result)
                print(f"{result['total_time']:.3f}s")

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    print("\n")

    return df


def compare_solvers(df):
    """Compare fixed-point vs Newton performance."""
    print("=" * 70)
    print("Solver Comparison (Fixed-Point vs Newton)")
    print("=" * 70)

    # Group by dimension and grid size
    for dim in df["dimension"].unique():
        df_dim = df[df["dimension"] == dim]

        if dim == 1:
            size_col = "Nx"
        else:
            size_col = "N"

        for size in df_dim[size_col].unique():
            df_subset = df_dim[df_dim[size_col] == size]

            # Compare solvers
            fp_time = df_subset[df_subset["solver_type"] == "fixed_point"]["total_time"].mean()
            newton_time = df_subset[df_subset["solver_type"] == "newton"]["total_time"].mean()

            speedup = newton_time / fp_time

            print(f"\n{dim}D, {size_col}={size}:")
            print(f"  Fixed-point: {fp_time:.3f}s")
            print(f"  Newton:      {newton_time:.3f}s")
            print(f"  Speedup (FP/Newton): {speedup:.2f}x")


def save_results(df_1d, df_2d, filename="benchmarks/results/hjb_solver_comparison.csv"):
    """Save results to CSV."""
    import os

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df_combined = pd.concat([df_1d, df_2d], ignore_index=True)
    df_combined.to_csv(filename, index=False)
    print(f"\n\nResults saved to: {filename}")


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print(" HJB SOLVER PERFORMANCE BENCHMARKS")
    print("*" * 70)
    print("\n")

    # Run benchmarks
    df_1d = run_1d_benchmarks()
    df_2d = run_2d_benchmarks()

    # Compare
    df_combined = pd.concat([df_1d, df_2d], ignore_index=True)
    compare_solvers(df_combined)

    # Save
    save_results(df_1d, df_2d)

    print("\n" + "*" * 70)
    print(" BENCHMARKS COMPLETE")
    print("*" * 70 + "\n")
