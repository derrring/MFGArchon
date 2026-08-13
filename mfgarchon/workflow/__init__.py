"""
Workflow Management System for MFGarchon

This module provides a comprehensive workflow management system for reproducible
research, experiment tracking, and collaborative development in mean field games.

Features:
- Reproducible experiment workflows with automatic versioning
- Parameter sweep and batch processing capabilities
- Result comparison and analysis across experiments
- Collaborative workspace management
- Automatic report generation and documentation
- Integration with version control and data provenance

Components:
- WorkflowManager: Core workflow orchestration
- ExperimentTracker: Experiment lifecycle management
- ParameterSweep: Automated parameter space exploration
- ResultAnalyzer: Cross-experiment analysis and comparison
- CollaborativeWorkspace: Multi-user research environment
- ReportGenerator: Automated documentation and visualization
"""

from __future__ import annotations

from typing import Any

from .common import ExecutionStatus
from .decorators import experiment, parameter_study, workflow_step
from .experiment_tracker import Experiment, ExperimentResult, ExperimentTracker
from .parameter_sweep import ParameterSweep, SweepConfiguration
from .workflow_manager import Workflow, WorkflowManager

# Version and compatibility
__version__ = "1.0.0"
__workflow_api_version__ = "1.0"

# Global workflow manager instance
# Thread-safe lazy singleton (Issue #759)
import threading

_global_workflow_manager: WorkflowManager | None = None
_workflow_manager_lock = threading.Lock()


def get_workflow_manager() -> WorkflowManager:
    """Get the global workflow manager instance (thread-safe)."""
    global _global_workflow_manager
    if _global_workflow_manager is None:
        with _workflow_manager_lock:
            # Double-check locking pattern
            if _global_workflow_manager is None:
                _global_workflow_manager = WorkflowManager()
    return _global_workflow_manager


def set_workflow_manager(manager: WorkflowManager) -> None:
    """Set the global workflow manager instance (thread-safe)."""
    global _global_workflow_manager
    with _workflow_manager_lock:
        _global_workflow_manager = manager


def create_workflow(name: str, description: str = "", **kwargs) -> Workflow:
    """Create a new workflow with the global manager."""
    return get_workflow_manager().create_workflow(name, description, **kwargs)


def create_experiment(name: str, workflow: Workflow | None = None, **kwargs) -> Experiment:
    """Create a new experiment."""
    tracker = ExperimentTracker()
    return tracker.create_experiment(name, workflow=workflow, **kwargs)


def create_parameter_sweep(parameters: dict[str, Any], **kwargs) -> ParameterSweep:
    """Create a parameter sweep configuration."""
    return ParameterSweep(parameters, **kwargs)


# Note: Advanced features like collaborative workspace, result analysis, and report generation
# are planned for future implementation


# Convenience functions for common workflows
def mfg_parameter_study(
    problem_factory,
    parameters: dict[str, list],
    solver_type: str = "fixed_point",
    **kwargs,
):
    """
    Convenient function for MFG parameter studies.

    Args:
        problem_factory: Function that creates MFG problems
        parameters: Dictionary of parameter names to lists of values
        solver_type: Type of solver to use
        **kwargs: Additional arguments for the workflow

    Returns:
        Workflow with parameter study configured
    """
    workflow = create_workflow(
        name=f"mfg_parameter_study_{solver_type}",
        description=f"Parameter study for MFG system using {solver_type} solver",
    )

    # Create parameter sweep
    sweep = create_parameter_sweep(parameters)

    # Define solve step
    @workflow_step(workflow)
    def solve_mfg_problem(params):
        problem = problem_factory(**params)

        # Use problem.solve() API
        return problem.solve()

    # Execute parameter sweep
    results = sweep.execute(solve_mfg_problem)

    return workflow, results


def convergence_analysis_workflow(problem, solver_types: list[str], tolerances: list[float], **kwargs):
    """
    Workflow for analyzing convergence across different solvers and tolerances.

    Args:
        problem: MFG problem instance
        solver_types: List of solver types to compare
        tolerances: List of tolerance values to test
        **kwargs: Additional workflow arguments

    Returns:
        Workflow with convergence analysis configured
    """
    workflow = create_workflow(
        name="convergence_analysis",
        description="Convergence analysis across solvers and tolerances",
    )

    # Create parameter combinations
    parameters = {"solver_type": solver_types, "tolerance": tolerances}

    sweep = create_parameter_sweep(parameters)

    @workflow_step(workflow)
    def analyze_convergence(params):
        # Use problem.solve() API with tolerance parameter
        result = problem.solve(tolerance=params["tolerance"])

        return {
            "solver_type": params["solver_type"],
            "tolerance": params["tolerance"],
            "converged": getattr(result, "converged", False),
            "iterations": getattr(result, "iterations", 0),
            "final_error": getattr(result, "final_error", 0.0),
            "execution_time": getattr(result, "execution_time", 0.0),
        }

    results = sweep.execute(analyze_convergence)

    return workflow, results


def performance_benchmark_workflow(problem_sizes: list[tuple], solver_types: list[str], **kwargs):
    """
    Workflow for performance benchmarking across problem sizes and solvers.

    Args:
        problem_sizes: List of (Nx, Nt) tuples
        solver_types: List of solver types to benchmark
        **kwargs: Additional workflow arguments

    Returns:
        Workflow with performance benchmarking configured
    """
    workflow = create_workflow(
        name="performance_benchmark",
        description="Performance benchmarking across problem sizes and solvers",
    )

    parameters = {"problem_size": problem_sizes, "solver_type": solver_types}

    sweep = create_parameter_sweep(parameters)

    @workflow_step(workflow)
    def benchmark_performance(params):
        import time

        from mfgarchon import MFGProblem
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import no_flux_bc

        Nx, Nt = params["problem_size"]
        geometry = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1)
        )
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=Nt)

        # Use problem.solve() API
        start_time = time.time()
        result = problem.solve()
        execution_time = time.time() - start_time

        return {
            "problem_size": (Nx, Nt),
            "solver_type": params["solver_type"],
            "grid_points": Nx * Nt,
            "execution_time": execution_time,
            "converged": getattr(result, "converged", False),
            "iterations": getattr(result, "iterations", 0),
            "memory_estimate": Nx * Nt * 16 / 1024**2,  # Rough estimate in MB
        }

    results = sweep.execute(benchmark_performance)

    return workflow, results


# Export public API
__all__ = [
    "ExecutionStatus",
    "Experiment",
    "ExperimentResult",
    "ExperimentTracker",
    "ParameterSweep",
    "SweepConfiguration",
    "Workflow",
    "WorkflowManager",
    "convergence_analysis_workflow",
    "create_experiment",
    "create_parameter_sweep",
    "create_workflow",
    "experiment",
    "get_workflow_manager",
    "mfg_parameter_study",
    "parameter_study",
    "performance_benchmark_workflow",
    "set_workflow_manager",
    "workflow_step",
]


# [REMOVED 2026-08-14] `_initialize_workflow_system()` ran at import and is gone. It built the
# global manager, created `.mfg_workflows/` under the CALLER's working directory, and persisted an
# example workflow -- `example_parameter_study`, with a step -- as a side effect of reading the
# module. The `MFG_WORKFLOW_INITIALIZED` env guard only suppressed a repeat within one process, so
# every process contributed a fresh UUID directory: 20,322 accumulated in this repository in two
# weeks, and a sample of 500 were all empty. Nothing in the survey that found them was using the
# workflow system; they came from test collection and from `import mfgarchon`.
#
# `initialize_default_workspace()` and `_create_example_workflows()` were unreachable once this
# call went, and are deleted with it -- the change's own tail. Example content belongs in
# `examples/`, not in a directory the library invents under someone else's cwd. #1917
