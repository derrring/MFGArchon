"""
Constraint protocols for variational inequality problems.

This module defines Protocol classes for inequality constraints in MFG systems,
enabling obstacle problems, capacity constraints, and congestion modeling.

Mathematical Background:
    Variational Inequality (VI): Find u such that:
        ⟨F(u), v - u⟩ ≥ 0  ∀v ∈ K

    where K is a convex constraint set defined by:
        - Obstacle constraint: K = {u : u ≥ ψ} (lower) or K = {u : u ≤ ψ} (upper)
        - Bilateral constraint: K = {u : ψ_lower ≤ u ≤ ψ_upper}

    Projection onto K: P_K(u) = argmin_{v ∈ K} ‖v - u‖

    Properties:
        - **Idempotent**: P_K(P_K(u)) = P_K(u)
        - **Non-expansive**: ‖P_K(u) - P_K(v)‖ ≤ ‖u - v‖
        - **Fixed point**: u ∈ K ⟺ P_K(u) = u

MFG Applications:
    1. **Capacity constraints**: m(t,x) ≤ m_max (crowd density limits)
    2. **HJB with obstacles**: u(t,x) ≥ ψ(x) (running cost floor)
    3. **Congestion pricing**: c(m) includes penalty for m > threshold
    4. **Free boundaries**: Stefan problems, interface tracking

References:
    - Glowinski et al. (1984): Numerical Methods for Nonlinear Variational Problems
    - Achdou & Capuzzo-Dolcetta (2010): Mean Field Games: Numerical Methods
    - Cottle et al. (2009): The Linear Complementarity Problem

Created: 2026-01-17 (Issue #591 - Phase 2.1: Constraint Protocols)
Part of: Issue #589 Phase 2 (Tier 2 BCs - Variational Constraints)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "ConstraintProtocol",
]


@runtime_checkable
class ConstraintProtocol(Protocol):
    """
    Protocol for inequality constraints in variational inequality problems.

    Constrains the VALUES a field may take. It has nothing to do with geometric obstacles
    (`problem.obstacles`, `obstacle_sdf`), which remove a region from the domain, nor with a
    soft wall (`problem.state_penalty`), which is a potential. `ObstacleConstraint`'s
    docstring has the full three-way disambiguation; it is not repeated here.

    Defines the interface for constraints that restrict the solution space
    to a convex set K ⊂ ℝ^N. Used in penalty methods, projected Newton, and
    active set algorithms for solving VIs.

    **Mathematical Context**:
        A constraint defines a convex set K and provides:
            1. Projection: P_K(u) = argmin_{v ∈ K} ‖v - u‖
            2. Feasibility test: u ∈ K?
            3. Active set: A(u) = {i : constraint binds at u_i}

    **Usage in VI Solvers**:
        >>> # Penalty method
        >>> for iteration in range(max_iterations):
        >>>     u_unconstrained = solve_unconstrained_step()
        >>>     u = constraint.project(u_unconstrained)  # Enforce constraint
        >>>
        >>> # Active set method
        >>> active = constraint.get_active_set(u, tol=1e-8)
        >>> # Solve reduced problem on inactive set

    **Properties Required**:
        - Projection is idempotent: project(project(u)) = project(u)
        - Projection is non-expansive: ‖P_K(u) - P_K(v)‖ ≤ ‖u - v‖
        - Feasibility consistent: is_feasible(project(u)) = True

    **Concrete Implementations**:
        - ObstacleConstraint: K = {u : u ≥ ψ} or {u : u ≤ ψ}
        - BilateralConstraint: K = {u : ψ_lower ≤ u ≤ ψ_upper}
        - RegionConstraint: K = {u : u satisfies constraints in region R}

    Example:
        >>> from mfgarchon.geometry.boundary.constraints import ObstacleConstraint
        >>> import numpy as np
        >>>
        >>> # Capacity constraint: crowd density ≤ 0.5
        >>> m_max = 0.5 * np.ones(100)
        >>> constraint = ObstacleConstraint(m_max, constraint_type='upper')
        >>>
        >>> # Check if density is feasible
        >>> m = np.random.rand(100)
        >>> if not constraint.is_feasible(m):
        >>>     m = constraint.project(m)  # Enforce capacity limit
        >>>
        >>> # Identify overcrowded locations
        >>> active = constraint.get_active_set(m, tol=1e-6)
        >>> print(f"{active.sum()} locations at capacity")
    """

    def project(self, u: NDArray) -> NDArray:
        """
        Project field onto constraint set K.

        Computes P_K(u) = argmin_{v ∈ K} ‖v - u‖.

        Args:
            u: Field to project, shape (N,) or (Nx, Ny, ...)

        Returns:
            Projected field P_K(u), same shape as u

        Properties:
            - **Idempotent**: project(project(u)) = project(u)
            - **Non-expansive**: ‖P_K(u) - P_K(v)‖ ≤ ‖u - v‖
            - **Feasible**: is_feasible(project(u)) = True

        Example:
            >>> # Obstacle constraint u ≥ ψ
            >>> u = np.array([1.0, -0.5, 2.0])
            >>> psi = np.array([0.5, 0.0, 0.5])
            >>> constraint = ObstacleConstraint(psi, constraint_type='lower')
            >>> u_proj = constraint.project(u)
            >>> # u_proj = [1.0, 0.0, 2.0]  (enforces u ≥ ψ)

        Note:
            For obstacle constraints, projection is trivial: max(u, ψ) or min(u, ψ).
            For general convex sets, may require optimization (e.g., Dykstra's algorithm).
        """
        ...

    def is_feasible(self, u: NDArray, tol: float = 1e-10) -> bool:
        """
        Check if field satisfies constraint.

        Tests whether u ∈ K within tolerance tol.

        Args:
            u: Field to check, shape (N,) or (Nx, Ny, ...)
            tol: Tolerance for constraint violation (default 1e-10)

        Returns:
            True if u ∈ K (within tol), False otherwise

        Example:
            >>> constraint = ObstacleConstraint(psi, constraint_type='lower')
            >>> u = np.array([1.0, 0.5, 2.0])
            >>> psi = np.array([0.5, 0.0, 0.5])
            >>> constraint.is_feasible(u)  # True (all u >= psi)
            >>>
            >>> u_bad = np.array([0.3, -0.1, 1.0])
            >>> constraint.is_feasible(u_bad)  # False (u[0], u[1] < psi)

        Note:
            This is primarily for debugging and validation. In production,
            use project() to enforce constraints rather than checking feasibility.
        """
        ...

    def get_active_set(self, u: NDArray, tol: float = 1e-10) -> NDArray:
        """
        Identify active constraints (where constraint binds).

        For obstacle constraint u ≥ ψ:
            Active set A = {i : u_i ≈ ψ_i} (binding)
            Inactive set I = {i : u_i > ψ_i + tol} (slack)

        Args:
            u: Field to analyze, shape (N,) or (Nx, Ny, ...)
            tol: Tolerance for detecting binding (default 1e-10)

        Returns:
            Boolean mask, same shape as u
                - True: constraint is active (binding) at this point
                - False: constraint is inactive (has slack)

        Example:
            >>> constraint = ObstacleConstraint(psi, constraint_type='lower')
            >>> u = np.array([1.0, 0.5, 0.5])  # Last two at obstacle
            >>> psi = np.array([0.5, 0.5, 0.5])
            >>> active = constraint.get_active_set(u, tol=1e-8)
            >>> # active = [False, True, True]

        Applications:
            - **Active set methods**: Solve reduced problem on inactive set
            - **Diagnostics**: Identify where constraints bind
            - **Convergence**: Monitor active set stability

        Note:
            For bilateral constraints (ψ_lower ≤ u ≤ ψ_upper), returns mask
            where EITHER lower or upper constraint is active.
        """
        ...


if __name__ == "__main__":
    """Smoke test for ConstraintProtocol."""
    import numpy as np

    print("Testing ConstraintProtocol...")

    # Create mock constraint for testing protocol
    class MockObstacleConstraint:
        """Mock implementation for protocol testing."""

        def __init__(self, obstacle: NDArray, constraint_type: str = "lower"):
            self.obstacle = obstacle
            self.constraint_type = constraint_type

        def project(self, u: NDArray) -> NDArray:
            if self.constraint_type == "lower":
                return np.maximum(u, self.obstacle)
            else:  # upper
                return np.minimum(u, self.obstacle)

        def is_feasible(self, u: NDArray, tol: float = 1e-10) -> bool:
            if self.constraint_type == "lower":
                return np.all(u >= self.obstacle - tol)
            else:  # upper
                return np.all(u <= self.obstacle + tol)

        def get_active_set(self, u: NDArray, tol: float = 1e-10) -> NDArray:
            if self.constraint_type == "lower":
                return np.abs(u - self.obstacle) < tol
            else:  # upper
                return np.abs(u - self.obstacle) < tol

    # Test protocol compliance
    print("\n[Protocol Compliance]")
    psi = np.array([0.5, 1.0, 0.0])
    constraint = MockObstacleConstraint(psi, constraint_type="lower")

    assert isinstance(constraint, ConstraintProtocol), "Should implement ConstraintProtocol"
    print("  ✓ MockObstacleConstraint implements ConstraintProtocol")

    # Test projection
    print("\n[Projection]")
    u = np.array([1.0, 0.5, -0.5])
    u_proj = constraint.project(u)
    expected = np.array([1.0, 1.0, 0.0])  # Enforces u >= psi

    print(f"  Input: {u}")
    print(f"  Obstacle: {psi}")
    print(f"  Projected: {u_proj}")
    print(f"  Expected: {expected}")
    assert np.allclose(u_proj, expected), "Projection failed"
    print("  ✓ Projection works")

    # Test idempotence
    u_proj_twice = constraint.project(u_proj)
    error = np.max(np.abs(u_proj_twice - u_proj))
    print(f"  Idempotence error: {error:.2e}")
    assert error < 1e-12, "Projection not idempotent"
    print("  ✓ Projection is idempotent")

    # Test feasibility
    print("\n[Feasibility]")
    assert not constraint.is_feasible(u), "Infeasible u should fail"
    assert constraint.is_feasible(u_proj), "Projected u should be feasible"
    print("  ✓ Feasibility check works")

    # Test active set
    print("\n[Active Set]")
    active = constraint.get_active_set(u_proj, tol=1e-8)
    expected_active = np.array([False, True, True])  # Last two at obstacle

    print(f"  Projected u: {u_proj}")
    print(f"  Obstacle: {psi}")
    print(f"  Active set: {active}")
    print(f"  Expected: {expected_active}")
    assert np.array_equal(active, expected_active), "Active set incorrect"
    print("  ✓ Active set detection works")

    print("\n✅ All ConstraintProtocol tests passed!")
