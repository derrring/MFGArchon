"""
MFG Solver Factory Module

Provides factory patterns for easy creation of problems and solver configurations.

Problem factories:
- create_mfg_problem() - Main factory for all problem types
- create_standard_problem() - Standard HJB-FP MFG
- create_network_problem() - Network/Graph MFG
- create_variational_problem() - Variational/Lagrangian MFG
- create_stochastic_problem() - Stochastic MFG with common noise
- create_highdim_problem() - High-dimensional MFG (d > 3)
- create_lq_problem() - Linear-Quadratic MFG
- create_crowd_problem() - Crowd dynamics MFG

Solver:
- SolverFactory - Main solver factory class
- create_solver() - Convenience function for creating solvers

For most use cases, prefer problem.solve() over create_solver().
"""

# Backend support
from .backend_factory import BackendFactory, create_backend_for_problem, print_backend_info

# Problem factories
from .problem_factories import (
    create_crowd_problem,
    create_highdim_problem,
    create_lq_problem,
    create_mfg_problem,
    create_network_problem,
    create_standard_problem,
    create_stochastic_problem,
    create_variational_problem,
    # Classic LQ-MFG conditions (Issue #670)
    lq_mfg_initial_density,
    lq_mfg_terminal_cost,
)

# Scheme-based paired solver factory (Issue #580)
from .scheme_factory import create_paired_solvers, get_recommended_scheme

# Solver factory
from .solver_factory import (
    SolverFactory,
    create_accurate_solver,
    create_basic_solver,
    create_fast_solver,
    create_research_solver,
    create_solver,
)

__all__ = [
    # Problem factories
    "create_mfg_problem",
    "create_standard_problem",
    "create_network_problem",
    "create_variational_problem",
    "create_stochastic_problem",
    "create_highdim_problem",
    "create_lq_problem",
    "create_crowd_problem",
    # Classic LQ-MFG conditions (Issue #670)
    "lq_mfg_initial_density",
    "lq_mfg_terminal_cost",
    # Solver
    "SolverFactory",
    "create_solver",
    # Deprecated solver factories (raise NotImplementedError)
    "create_fast_solver",
    "create_accurate_solver",
    "create_research_solver",
    "create_basic_solver",
    # Scheme-based paired solver factory (Issue #580)
    "create_paired_solvers",
    "get_recommended_scheme",
    # Backend support
    "BackendFactory",
    "create_backend_for_problem",
    "print_backend_info",
    # General purpose factory
    "create_general_mfg_problem",
]
