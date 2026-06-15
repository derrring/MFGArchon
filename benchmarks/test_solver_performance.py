#!/usr/bin/env python3
"""
Performance Regression Tests for MFG Solvers

This module provides automated performance benchmarking for solver regression testing.
Uses pytest-benchmark to track solver performance across code changes.

Usage:
    # Run benchmarks only
    pytest benchmarks/test_solver_performance.py --benchmark-only

    # Save baseline
    pytest benchmarks/test_solver_performance.py --benchmark-only --benchmark-save=baseline

    # Compare against baseline
    pytest benchmarks/test_solver_performance.py --benchmark-only --benchmark-compare=baseline

    # Generate detailed report
    pytest benchmarks/test_solver_performance.py --benchmark-only --benchmark-histogram
"""

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.mfg_solvers import ParticleCollocationSolver
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


# Test problem fixtures
@pytest.fixture
def small_problem():
    """Small problem for quick regression testing (suitable for CI)."""
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        T=1.0,
        Nt=30,
        sigma=0.12,
        coupling_coefficient=0.02,
    )


@pytest.fixture
def medium_problem():
    """Medium problem for comprehensive benchmarking."""
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[40 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        T=1.0,
        Nt=50,
        sigma=0.12,
        coupling_coefficient=0.02,
    )


# Solver benchmarks
@pytest.mark.benchmark(group="solvers-small")
def test_particle_collocation_small_performance(benchmark, small_problem):
    """
    Baseline: ParticleCollocationSolver on small problem.

    This test establishes a performance baseline for the unified
    ParticleCollocationSolver on a small grid suitable for CI.
    """
    # Create collocation points (required parameter)
    bounds = small_problem.geometry.get_bounds()
    Nx_points = small_problem.geometry.get_grid_shape()[0]
    x_collocation = np.linspace(bounds[0][0], bounds[1][0], Nx_points)

    solver = ParticleCollocationSolver(
        problem=small_problem,
        collocation_points=x_collocation.reshape(-1, 1),  # Must be 2D array
        num_particles=500,  # Moderate particle count
    )

    # Benchmark the solve method
    _, _, metadata = benchmark(solver.solve)

    # Verify convergence
    assert metadata.get("converged", False), "Solver must converge for valid benchmark"
    assert metadata.get("iterations", 0) > 0, "Solver must perform iterations"


@pytest.mark.benchmark(group="solvers-medium")
@pytest.mark.slow
def test_particle_collocation_medium_performance(benchmark, medium_problem):
    """
    Extended: ParticleCollocationSolver on medium problem.

    This test provides more comprehensive performance data but takes
    longer to run. Marked as 'slow' for optional execution.
    """
    # Create collocation points (required parameter)
    bounds = medium_problem.geometry.get_bounds()
    Nx_points = medium_problem.geometry.get_grid_shape()[0]
    x_collocation = np.linspace(bounds[0][0], bounds[1][0], Nx_points)

    solver = ParticleCollocationSolver(
        problem=medium_problem,
        collocation_points=x_collocation.reshape(-1, 1),  # Must be 2D array
        num_particles=1000,  # More particles for accuracy
    )

    # Benchmark the solve method
    _, _, metadata = benchmark(solver.solve)

    # Verify convergence
    assert metadata.get("converged", False), "Solver must converge for valid benchmark"


# Factory pattern benchmarks
@pytest.mark.benchmark(group="factory")
def test_solver_creation_overhead(benchmark):
    """
    Benchmark solver instantiation overhead.

    Measures the time to create solver instances, which should be
    negligible compared to solve time.
    """
    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=30,
    )
    bounds = problem.geometry.get_bounds()
    Nx_points = problem.geometry.get_grid_shape()[0]
    x_collocation = np.linspace(bounds[0][0], bounds[1][0], Nx_points)

    def create_solver():
        return ParticleCollocationSolver(
            problem=problem,
            collocation_points=x_collocation.reshape(-1, 1),
            num_particles=500,
        )

    solver = benchmark(create_solver)
    assert solver is not None


# Problem creation benchmarks
@pytest.mark.benchmark(group="problem-creation")
def test_problem_creation_overhead(benchmark):
    """
    Benchmark problem instantiation overhead.

    Ensures that problem setup time is negligible.
    """

    def create_problem():
        return MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[40 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            T=1.0,
            Nt=50,
            sigma=0.12,
            coupling_coefficient=0.02,
        )

    problem = benchmark(create_problem)
    assert problem is not None
    Nx_intervals = problem.geometry.get_grid_shape()[0] - 1
    assert Nx_intervals == 40
    assert problem.Nt == 50


# Configuration notes for CI integration
"""
Recommended pytest-benchmark configuration for CI (.benchmarks config):

{
    "disable_gc": true,
    "warmup": true,
    "warmup_iterations": 2,
    "min_rounds": 5,
    "max_time": 1.0,
    "calibration_precision": 10,
    "timer": "time.perf_counter"
}

For CI workflow, use:
    --benchmark-only                    # Skip non-benchmark tests
    --benchmark-min-rounds=3            # Fewer rounds for faster CI
    --benchmark-json=results.json       # Save results for comparison
    --benchmark-compare=baseline        # Compare against baseline
    --benchmark-compare-fail=mean:10%   # Fail if >10% slower
"""
