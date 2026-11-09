"""
MFG Solver Factory Module

Provides factory patterns for easy creation of problems and solver configurations.

Problem data:
- lq_mfg_initial_density(), lq_mfg_terminal_cost() - the classic LQ-MFG conditions

Solver:
- SolverFactory - Main solver factory class

For most use cases, prefer problem.solve().
"""

# Backend support
from .backend_factory import BackendFactory, create_backend_for_problem, print_backend_info

# Classic LQ-MFG conditions (Issue #670)
from .problem_factories import (
    lq_mfg_initial_density,
    lq_mfg_terminal_cost,
)

# Scheme-based paired solver factory (Issue #580)
from .scheme_factory import create_paired_solvers, get_recommended_scheme

# Solver factory
from .solver_factory import SolverFactory

__all__ = [
    # Classic LQ-MFG conditions (Issue #670)
    "lq_mfg_initial_density",
    "lq_mfg_terminal_cost",
    # Solver
    "SolverFactory",
    # Scheme-based paired solver factory (Issue #580)
    "create_paired_solvers",
    "get_recommended_scheme",
    # Backend support
    "BackendFactory",
    "create_backend_for_problem",
    "print_backend_info",
]
