"""
Performance benchmarks for callable coefficient evaluation.

Compares performance of:
- Scalar diffusion (baseline)
- Array diffusion (spatial/spatiotemporal)
- Callable diffusion (state-dependent)

Measures overhead introduced by CoefficientField abstraction and
callable evaluation at each timestep.
"""

from __future__ import annotations

import time

import numpy as np

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def benchmark_scalar_diffusion(Nx=100, Nt=100, num_runs=3):
    """Benchmark MFG with constant scalar diffusion (baseline)."""
    times = []

    for _ in range(num_runs):
        # Create problem with scalar diffusion
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            problem = MFGProblem(
                geometry=TensorProductGrid(
                    bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1)
                ),
                T=1.0,
                Nt=Nt,
                sigma=0.1,
                drift_weight=1.0,
                coupling_lambda=1.0,
            )

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(problem, hjb_solver, fp_solver, damping_factor=0.5)

        start = time.perf_counter()
        mfg_solver.solve(max_iterations=5, tolerance=1e-6, verbose=False)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return {"mean": np.mean(times), "std": np.std(times), "min": np.min(times), "max": np.max(times), "runs": num_runs}


def benchmark_array_diffusion_spatial(Nx=100, Nt=100, num_runs=3):
    """Benchmark MFG with spatially varying array diffusion."""
    times = []

    for _ in range(num_runs):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            problem = MFGProblem(
                geometry=TensorProductGrid(
                    bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1)
                ),
                T=1.0,
                Nt=Nt,
                sigma=0.1,
                drift_weight=1.0,
                coupling_lambda=1.0,
            )

        # Create spatially varying diffusion
        x = np.linspace(0, 1, Nx + 1)
        sigma_array = 0.08 + 0.04 * x  # Increases with x

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(problem, hjb_solver, fp_solver, damping_factor=0.5, diffusion_field=sigma_array)

        start = time.perf_counter()
        mfg_solver.solve(max_iterations=5, tolerance=1e-6, verbose=False)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return {"mean": np.mean(times), "std": np.std(times), "min": np.min(times), "max": np.max(times), "runs": num_runs}


def benchmark_array_diffusion_spatiotemporal(Nx=100, Nt=100, num_runs=3):
    """Benchmark MFG with spatiotemporal array diffusion."""
    times = []

    for _ in range(num_runs):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            problem = MFGProblem(
                geometry=TensorProductGrid(
                    bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1)
                ),
                T=1.0,
                Nt=Nt,
                sigma=0.1,
                drift_weight=1.0,
                coupling_lambda=1.0,
            )

        # Create spatiotemporal diffusion
        x = np.linspace(0, 1, Nx + 1)
        t = np.linspace(0, 1, Nt + 1)
        X, T = np.meshgrid(x, t)
        sigma_array = 0.08 + 0.02 * (X + T)  # Varies in space and time

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(problem, hjb_solver, fp_solver, damping_factor=0.5, diffusion_field=sigma_array)

        start = time.perf_counter()
        mfg_solver.solve(max_iterations=5, tolerance=1e-6, verbose=False)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return {"mean": np.mean(times), "std": np.std(times), "min": np.min(times), "max": np.max(times), "runs": num_runs}


def benchmark_callable_diffusion_scalar(Nx=100, Nt=100, num_runs=3):
    """Benchmark MFG with callable diffusion returning scalar."""
    times = []

    def constant_diffusion(t, x, m):
        return 0.1

    for _ in range(num_runs):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            problem = MFGProblem(
                geometry=TensorProductGrid(
                    bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1)
                ),
                T=1.0,
                Nt=Nt,
                sigma=0.1,
                drift_weight=1.0,
                coupling_lambda=1.0,
            )

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(
            problem, hjb_solver, fp_solver, damping_factor=0.5, diffusion_field=constant_diffusion
        )

        start = time.perf_counter()
        mfg_solver.solve(max_iterations=5, tolerance=1e-6, verbose=False)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return {"mean": np.mean(times), "std": np.std(times), "min": np.min(times), "max": np.max(times), "runs": num_runs}


def benchmark_callable_diffusion_porous_medium(Nx=100, Nt=100, num_runs=3):
    """Benchmark MFG with porous medium callable diffusion."""
    times = []

    def porous_medium(t, x, m):
        return 0.05 * m

    for _ in range(num_runs):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            problem = MFGProblem(
                geometry=TensorProductGrid(
                    bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1)
                ),
                T=1.0,
                Nt=Nt,
                sigma=0.1,
                drift_weight=1.0,
                coupling_lambda=1.0,
            )

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(
            problem, hjb_solver, fp_solver, damping_factor=0.5, diffusion_field=porous_medium
        )

        start = time.perf_counter()
        mfg_solver.solve(max_iterations=5, tolerance=1e-6, verbose=False)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return {"mean": np.mean(times), "std": np.std(times), "min": np.min(times), "max": np.max(times), "runs": num_runs}


def benchmark_callable_diffusion_crowd_dynamics(Nx=100, Nt=100, num_runs=3):
    """Benchmark MFG with crowd dynamics callable diffusion."""
    times = []

    def crowd_diffusion(t, x, m):
        m_max = np.max(m) if np.max(m) > 0 else 1.0
        return 0.05 + 0.05 * (1 - m / m_max)

    for _ in range(num_runs):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            problem = MFGProblem(
                geometry=TensorProductGrid(
                    bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1)
                ),
                T=1.0,
                Nt=Nt,
                sigma=0.1,
                drift_weight=1.0,
                coupling_lambda=1.0,
            )

        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        mfg_solver = FixedPointIterator(
            problem, hjb_solver, fp_solver, damping_factor=0.5, diffusion_field=crowd_diffusion
        )

        start = time.perf_counter()
        mfg_solver.solve(max_iterations=5, tolerance=1e-6, verbose=False)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return {"mean": np.mean(times), "std": np.std(times), "min": np.min(times), "max": np.max(times), "runs": num_runs}


def print_benchmark_results(results, baseline_time=None):
    """Print benchmark results in formatted table."""
    print(f"  Mean: {results['mean']:.4f}s ± {results['std']:.4f}s")
    print(f"  Range: [{results['min']:.4f}s, {results['max']:.4f}s]")

    if baseline_time is not None:
        overhead = ((results["mean"] - baseline_time) / baseline_time) * 100
        slowdown = results["mean"] / baseline_time
        print(f"  Overhead: {overhead:+.1f}% ({slowdown:.2f}x)")


def run_benchmarks(problem_sizes=None):
    """Run all benchmarks with different problem sizes."""
    if problem_sizes is None:
        problem_sizes = [(50, 50), (100, 100), (200, 100)]

    for Nx, Nt in problem_sizes:
        print(f"\n{'=' * 70}")
        print(f"Benchmark: Nx={Nx}, Nt={Nt}")
        print(f"{'=' * 70}")

        print("\n1. Scalar Diffusion (baseline):")
        baseline = benchmark_scalar_diffusion(Nx=Nx, Nt=Nt, num_runs=3)
        print_benchmark_results(baseline)
        baseline_time = baseline["mean"]

        print("\n2. Array Diffusion (spatial):")
        array_spatial = benchmark_array_diffusion_spatial(Nx=Nx, Nt=Nt, num_runs=3)
        print_benchmark_results(array_spatial, baseline_time)

        print("\n3. Array Diffusion (spatiotemporal):")
        array_st = benchmark_array_diffusion_spatiotemporal(Nx=Nx, Nt=Nt, num_runs=3)
        print_benchmark_results(array_st, baseline_time)

        print("\n4. Callable Diffusion (scalar return):")
        callable_scalar = benchmark_callable_diffusion_scalar(Nx=Nx, Nt=Nt, num_runs=3)
        print_benchmark_results(callable_scalar, baseline_time)

        print("\n5. Callable Diffusion (porous medium):")
        callable_pm = benchmark_callable_diffusion_porous_medium(Nx=Nx, Nt=Nt, num_runs=3)
        print_benchmark_results(callable_pm, baseline_time)

        print("\n6. Callable Diffusion (crowd dynamics):")
        callable_crowd = benchmark_callable_diffusion_crowd_dynamics(Nx=Nx, Nt=Nt, num_runs=3)
        print_benchmark_results(callable_crowd, baseline_time)


if __name__ == "__main__":
    print("Callable Coefficient Performance Benchmarks")
    print("=" * 70)
    print("\nComparing performance overhead of different coefficient types.")
    print("Each benchmark runs 5 MFG Picard iterations with 3 repetitions.")

    # Run benchmarks with different problem sizes
    run_benchmarks(
        problem_sizes=[
            (50, 50),  # Small: ~2500 DOF
            (100, 100),  # Medium: ~10000 DOF
            (200, 100),  # Large: ~20000 DOF
        ]
    )

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)
    print("\nKey Insights:")
    print("- Scalar baseline: Fastest (no array indexing)")
    print("- Array spatial: Small overhead (constant-time indexing)")
    print("- Array spatiotemporal: Slightly higher (2D indexing)")
    print("- Callable scalar: Function call overhead")
    print("- Callable state-dependent: Additional computation overhead")
    print("\nTarget: <2x slowdown for callable vs scalar (Phase 2 goal)")
