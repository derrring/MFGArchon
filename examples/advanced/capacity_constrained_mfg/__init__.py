"""
Capacity-constrained MFG for maze navigation and crowd dynamics.

This example demonstrates how to extend the MFGarchon framework with application-specific
capacity constraints for modeling congestion effects in maze environments.

Key Components:
    - CapacityField: Spatially-varying capacity C(x) from maze geometry
    - CongestionModel: Convex cost functions g(ρ) with derivatives
    - CapacityConstrainedMFGProblem: MFG problem with congestion term

Mathematical Framework:
    The Hamiltonian is extended with a congestion cost term:

    H(x, m, ∇u) = (1/2)|∇u|² + α·m + γ·g(m(x)/C(x))

    where m(x)/C(x) is the congestion ratio and g() is a convex penalty function.
    As density approaches capacity, the cost g(m/C) → ∞ creates a "soft wall" effect.

Examples:
    >>> from examples.advanced.capacity_constrained_mfg import (
    ...     CapacityField,
    ...     QuadraticCongestion,
    ...     CapacityConstrainedMFGProblem
    ... )
    >>> from mfgarchon.geometry.graph import MazeGeometry
    >>>
    >>> # Generate maze (MazeGeometry generates on construction)
    >>> maze = MazeGeometry(rows=20, cols=20, seed=42)
    >>> maze_array = maze.to_numpy_array(wall_thickness=3)
    >>>
    >>> # Compute capacity field
    >>> capacity = CapacityField.from_maze_geometry(maze_array)
    >>>
    >>> # Create capacity-constrained problem
    >>> problem = CapacityConstrainedMFGProblem(
    ...     capacity_field=capacity,
    ...     congestion_model=QuadraticCongestion(),
    ...     congestion_weight=1.0,
    ...     spatial_bounds=[(0, 1), (0, 1)],
    ...     spatial_discretization=[63, 63],
    ...     T=1.0,
    ...     Nt=50,
    ...     diffusion=0.01,
    ... )
    >>>
    >>> # Solve with any MFG solver
    >>> from mfgarchon.factory import create_fast_solver
    >>> solver = create_fast_solver(problem)
    >>> result = solver.solve()

References:
    - Hughes, R. L. (2002). "A continuum theory for the flow of pedestrians."
      Transportation Research Part B, 36(6), 507-535.
    - Achdou, Y., et al. (2020). "Mean field games with congestion."
      Annales de l'IHP Analyse non linéaire, 37(3), 637-663.

Created: 2025-11-12
"""

from .capacity_field import CapacityField, visualize_capacity_field
from .congestion import (
    CongestionModel,
    ExponentialCongestion,
    LogBarrierCongestion,
    PiecewiseCongestion,
    QuadraticCongestion,
    create_congestion_model,
)
from .problem import CapacityConstrainedMFGProblem

__all__ = [
    # Core problem class
    "CapacityConstrainedMFGProblem",
    # Capacity field
    "CapacityField",
    "visualize_capacity_field",
    # Congestion models
    "CongestionModel",
    "QuadraticCongestion",
    "ExponentialCongestion",
    "LogBarrierCongestion",
    "PiecewiseCongestion",
    "create_congestion_model",
]
