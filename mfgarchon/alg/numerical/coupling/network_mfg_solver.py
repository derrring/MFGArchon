"""
Network MFG Solver Factory.

This module provides factory functions for creating MFG solvers specialized
for network/graph structures, combining NetworkHJBSolver and FPNetworkSolver
with fixed-point iteration.

Mathematical Framework:
- Network HJB: -∂u/∂t + H_i(m, ∇_G u, t) = 0
- Network FP: ∂m/∂t - div_G(m ∇_G H_p) - σ²/2 Δ_G m = 0
- Fixed-point coupling: Iterate between HJB and FP until convergence

Key Features:
- Multiple solver schemes (explicit, implicit, flow-based)
- Graph-specific operators (Laplacian, gradients, divergence)
- Mass conservation on networks
- CFL stability for explicit schemes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.network_solvers.fp_network import FPNetworkSolver
from mfgarchon.alg.numerical.network_solvers.hjb_network import NetworkHJBSolver
from mfgarchon.utils.deprecation import deprecated_parameter

if TYPE_CHECKING:
    from mfgarchon.extensions.topology import NetworkMFGProblem


@deprecated_parameter(param_name="damping_factor", since="v0.19.2", replacement="relaxation")
def create_network_mfg_solver(
    problem: NetworkMFGProblem,
    solver_type: str = "fixed_point",
    hjb_solver_type: str | type = "RK45",
    fp_solver_type: str = "explicit",  # FP network solver not yet refactored
    relaxation: float = 0.5,
    hjb_kwargs: dict[str, Any] | None = None,
    fp_kwargs: dict[str, Any] | None = None,
    # Legacy kwarg (deprecated since v0.19.2, removal v0.25.0)
    damping_factor: float | None = None,
    **solver_kwargs,
) -> FixedPointIterator:
    """
    Create a complete MFG solver for network problems.

    Args:
        problem: NetworkMFGProblem instance
        solver_type: Type of MFG solver ("fixed_point" currently supported)
        hjb_solver_type: HJB scheme ("explicit", "implicit", "semi_implicit")
        fp_solver_type: FP scheme ("explicit", "implicit", "upwind", "flow")
        relaxation: Fixed-point iteration under-relaxation factor (0.5 = balanced)
        hjb_kwargs: Additional arguments for NetworkHJBSolver
        fp_kwargs: Additional arguments for FPNetworkSolver
        **solver_kwargs: Additional arguments for FixedPointIterator
            - use_anderson: bool = False (Anderson acceleration)
            - anderson_depth: int = 5 (Anderson memory depth)
            - backend: str = None (computational backend)

        Legacy `damping_factor` kwarg still accepted with DeprecationWarning.

    Returns:
        Configured MFG solver for network problems

    Example:
        >>> from mfgarchon import NetworkMFGProblem
        >>> from mfgarchon.alg.numerical.coupling.network_mfg_solver import create_network_mfg_solver
        >>> # Create problem (network geometry, components, etc.)
        >>> problem = NetworkMFGProblem(...)
        >>> # Create solver
        >>> solver = create_network_mfg_solver(
        ...     problem,
        ...     hjb_solver_type="implicit",
        ...     fp_solver_type="implicit",
        ...     relaxation=0.7,
        ... )
        >>> # Solve
        >>> U, M, info = solver.solve(max_iterations=50, tolerance=1e-6)
    """
    # Redirect legacy kwarg -> canonical (decorator already warned)
    if damping_factor is not None:
        relaxation = damping_factor

    # Default kwargs
    hjb_kwargs = hjb_kwargs or {}
    fp_kwargs = fp_kwargs or {}

    # Create HJB solver
    hjb_solver = NetworkHJBSolver(
        problem=problem,
        scheme=hjb_solver_type,
        **hjb_kwargs,
    )

    # Create FP solver
    fp_solver = FPNetworkSolver(
        problem=problem,
        scheme=fp_solver_type,
        **fp_kwargs,
    )

    # Create fixed-point iterator
    if solver_type == "fixed_point":
        solver = FixedPointIterator(
            problem=problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            relaxation=relaxation,
            **solver_kwargs,
        )
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}. Only 'fixed_point' is currently supported.")

    return solver


@deprecated_parameter(param_name="damping", since="v0.19.2", replacement="relaxation")
def create_simple_network_solver(
    problem: NetworkMFGProblem,
    scheme: str | type = "RK45",
    relaxation: float = 0.5,
    # Legacy kwarg (deprecated since v0.19.2, removal v0.25.0)
    damping: float | None = None,
) -> FixedPointIterator:
    """
    Create a simple network MFG solver with minimal configuration.

    Args:
        problem: NetworkMFGProblem instance
        scheme: Any scipy solve_ivp method for HJB. FP uses "explicit".
        relaxation: Fixed-point under-relaxation factor.

        Legacy `damping` kwarg still accepted with DeprecationWarning.

    Returns:
        Configured network MFG solver

    Example:
        >>> solver = create_simple_network_solver(problem)
        >>> U, M, info = solver.solve()
    """
    if damping is not None:
        relaxation = damping
    return create_network_mfg_solver(
        problem=problem,
        hjb_solver_type=scheme,
        fp_solver_type="explicit",  # FP network solver not yet refactored
        relaxation=relaxation,
    )


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing Network MFG Factory Functions...")

    # Test function availability
    assert create_network_mfg_solver is not None
    assert create_simple_network_solver is not None
    print("  Factory functions available")

    # Full smoke test requires NetworkMFGProblem setup
    # See examples/networks/ for usage examples

    print("Smoke tests passed!")
