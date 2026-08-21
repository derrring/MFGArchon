"""
Constraint implementations for variational inequality problems.

This module provides concrete implementations of the ConstraintProtocol for
obstacle problems, bilateral constraints, and capacity constraints in MFG systems.

"Obstacle" throughout this module means the VARIATIONAL INEQUALITY, never a region removed
from the domain -- two unrelated senses that meet only in the name `ObstacleConstraint`, whose
docstring carries the full disambiguation. Stated there once rather than here as well.

Mathematical Background:
    Variational Inequality (VI): Find u ∈ K such that:
        ⟨F(u), v - u⟩ ≥ 0  ∀v ∈ K

    where K is a convex constraint set. This module implements:

    1. **Obstacle Constraint** (Unilateral):
       - Lower: K = {u : u ≥ ψ}  →  P_K(u) = max(u, ψ)
       - Upper: K = {u : u ≤ ψ}  →  P_K(u) = min(u, ψ)

    2. **Bilateral Constraint** (Box):
       - K = {u : ψ_lower ≤ u ≤ ψ_upper}
       - P_K(u) = clip(u, ψ_lower, ψ_upper)

    Properties of Projection P_K:
        - **Idempotent**: P_K(P_K(u)) = P_K(u)
        - **Non-expansive**: ‖P_K(u) - P_K(v)‖ ≤ ‖u - v‖
        - **Fixed point**: u ∈ K ⟺ P_K(u) = u

MFG Applications:
    - **Capacity constraints**: m(t,x) ≤ m_max (crowd density limits)
    - **HJB obstacles**: u(t,x) ≥ ψ(x) (running cost floor)
    - **Congestion pricing**: Penalty when m > threshold
    - **Free boundaries**: Stefan problems, interface tracking

References:
    - Glowinski et al. (1984): Numerical Methods for Nonlinear Variational Problems
    - Achdou & Capuzzo-Dolcetta (2010): Mean Field Games: Numerical Methods
    - Kinderlehrer & Stampacchia (2000): An Introduction to Variational Inequalities

Created: 2026-01-17 (Issue #591 - Phase 2.1: Constraint Implementations)
Part of: Issue #589 Phase 2 (Tier 2 BCs - Variational Constraints)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "ObstacleConstraint",
    "BilateralConstraint",
]


class ObstacleConstraint:
    """
    Unilateral constraint for obstacle problems: u ≥ ψ or u ≤ ψ.

    **"Obstacle" here is the VARIATIONAL INEQUALITY, not a region of the domain.** The word
    carries two unrelated meanings in this package and they meet only in this class name:

    - *Obstacle problem* (this class) constrains the VALUES a field may take -- ``u ≥ ψ``, with
      a free boundary between the contact set and the continuation set. ψ is a level-set
      function on the same grid as ``u``; nothing is removed from the domain.
    - *Geometric obstacle* (``problem.obstacles``, ``obstacle_sdf``, maze walls, the GFDM
      stencil-visibility machinery) removes a REGION from the domain. Nothing there constrains
      the value of anything.

    The name is kept because "obstacle problem" is the standard term for this variational
    inequality, but a reader arriving from ``MFGProblem(obstacles=[Hypersphere(...)])`` -- the
    same package, a nearly identical word -- will read it the other way. Hence this paragraph.

    A third thing that is NEITHER: a soft wall, i.e. a cost for being somewhere. That is
    ``alpha``-free and ``u``-free, so it is a POTENTIAL, and it lives at
    ``problem.state_penalty``, composed into the Hamiltonian's V. It was called ``obstacle``
    until #2002, where it was documented as this class's inequality and implemented as a
    position penalty; that field now raises.

    The orthography, stated once: the bare noun ``obstacle*`` is geometry, the ``*Constraint``
    suffix is the variational inequality, and the potential has a name containing neither.

    Implements ConstraintProtocol for variational inequalities with single-sided
    bounds, common in HJB equations with running cost floors and capacity constraints
    in Fokker-Planck equations.

    **Mathematical Context**:
        - Lower obstacle: K = {u : u ≥ ψ}  (value function ≥ floor)
        - Upper obstacle: K = {u : u ≤ ψ}  (density ≤ capacity)

        Projection (trivial for obstacle constraints):
            - Lower: P_K(u) = max(u, ψ)  (enforce lower bound)
            - Upper: P_K(u) = min(u, ψ)  (enforce upper bound)

    **MFG Applications**:
        1. **Capacity-constrained MFG**: m(t,x) ≤ m_max
           >>> m_max = 0.5 * np.ones(grid_shape)
           >>> constraint = ObstacleConstraint(m_max, constraint_type='upper')

        2. **HJB with running cost floor**: u(t,x) ≥ ψ(x)
           >>> psi = compute_running_cost_floor(grid)
           >>> constraint = ObstacleConstraint(psi, constraint_type='lower')

        3. **Congestion pricing**: Penalty when m > threshold
           >>> threshold = 0.3 * np.ones(grid_shape)
           >>> constraint = ObstacleConstraint(threshold, constraint_type='upper')

    **Projection Properties**:
        - **Idempotent**: project(project(u)) = project(u)  ✓ (trivial for max/min)
        - **Non-expansive**: ‖P_K(u) - P_K(v)‖ ≤ ‖u - v‖  ✓
        - **Feasibility**: is_feasible(project(u)) = True  ✓

    Attributes:
        obstacle: Obstacle field ψ, shape (Nx, Ny, ...)
        constraint_type: 'lower' (u ≥ ψ) or 'upper' (u ≤ ψ)
        region: Optional mask for regional constraints, shape (Nx, Ny, ...)
                True = constrained, False = free

    Example:
        >>> import numpy as np
        >>> from mfgarchon.geometry.boundary.constraints import ObstacleConstraint
        >>>
        >>> # Capacity constraint: crowd density ≤ 0.5
        >>> grid_shape = (100, 100)
        >>> m_max = 0.5 * np.ones(grid_shape)
        >>> constraint = ObstacleConstraint(m_max, constraint_type='upper')
        >>>
        >>> # Check if density is feasible
        >>> m = 0.6 * np.ones(grid_shape)  # Exceeds capacity
        >>> assert not constraint.is_feasible(m)
        >>>
        >>> # Enforce capacity limit
        >>> m_feasible = constraint.project(m)
        >>> assert constraint.is_feasible(m_feasible)
        >>> assert np.allclose(m_feasible, 0.5)  # Clipped to capacity
        >>>
        >>> # Identify overcrowded locations
        >>> active = constraint.get_active_set(m_feasible)
        >>> print(f"{active.sum()} / {active.size} locations at capacity")

    Note:
        For bilateral constraints (box constraints), use BilateralConstraint instead.
    """

    def __init__(
        self,
        obstacle: NDArray,
        constraint_type: Literal["lower", "upper"] = "lower",
        region: NDArray | None = None,
    ):
        """
        Initialize obstacle constraint.

        Args:
            obstacle: Obstacle field ψ, shape (Nx, Ny, ...)
            constraint_type: Type of constraint
                - 'lower': u ≥ ψ (enforce minimum, e.g., cost floor)
                - 'upper': u ≤ ψ (enforce maximum, e.g., capacity limit)
            region: Optional boolean mask, same shape as obstacle
                - True: Apply constraint in this region
                - False: Free (no constraint)
                - None: Apply constraint everywhere (default)

        Raises:
            ValueError: If constraint_type not in {'lower', 'upper'}
            ValueError: If region shape doesn't match obstacle shape

        Example:
            >>> # Lower obstacle: u ≥ 0 (non-negative value function)
            >>> psi = np.zeros((50, 50))
            >>> constraint = ObstacleConstraint(psi, constraint_type='lower')
            >>>
            >>> # Upper obstacle with regional constraint
            >>> m_max = 0.5 * np.ones((50, 50))
            >>> bottleneck = np.zeros((50, 50), dtype=bool)
            >>> bottleneck[20:30, 20:30] = True  # Only constrain bottleneck region
            >>> constraint = ObstacleConstraint(m_max, constraint_type='upper', region=bottleneck)
        """
        if constraint_type not in ("lower", "upper"):
            raise ValueError(f"constraint_type must be 'lower' or 'upper', got '{constraint_type}'")

        self.obstacle = obstacle
        self.constraint_type = constraint_type
        self.region = region

        # Validate region shape
        if region is not None:
            if region.shape != obstacle.shape:
                raise ValueError(f"region shape {region.shape} doesn't match obstacle shape {obstacle.shape}")

    def project(self, u: NDArray) -> NDArray:
        """
        Project field onto constraint set K.

        Computes P_K(u) = argmin_{v ∈ K} ‖v - u‖.

        For obstacle constraints, projection is trivial:
            - Lower (u ≥ ψ): P_K(u) = max(u, ψ)
            - Upper (u ≤ ψ): P_K(u) = min(u, ψ)

        Args:
            u: Field to project, shape (Nx, Ny, ...)

        Returns:
            Projected field P_K(u), same shape as u

        Properties:
            - **Idempotent**: project(project(u)) = project(u)
            - **Non-expansive**: ‖P_K(u) - P_K(v)‖ ≤ ‖u - v‖
            - **Feasible**: is_feasible(project(u)) = True

        Example:
            >>> # Lower obstacle u ≥ ψ
            >>> u = np.array([1.0, -0.5, 2.0])
            >>> psi = np.array([0.5, 0.0, 0.5])
            >>> constraint = ObstacleConstraint(psi, constraint_type='lower')
            >>> u_proj = constraint.project(u)
            >>> # u_proj = [1.0, 0.0, 2.0]  (enforces u ≥ ψ)

            >>> # Upper obstacle u ≤ ψ (capacity limit)
            >>> m = np.array([0.3, 0.7, 0.4])
            >>> m_max = np.array([0.5, 0.5, 0.5])
            >>> constraint = ObstacleConstraint(m_max, constraint_type='upper')
            >>> m_proj = constraint.project(m)
            >>> # m_proj = [0.3, 0.5, 0.4]  (enforces m ≤ m_max)

        Note:
            If region mask is provided, projection only applied in constrained region.
            Free region values are unchanged.
        """
        # Validate shape
        if u.shape != self.obstacle.shape:
            raise ValueError(f"Field shape {u.shape} doesn't match obstacle shape {self.obstacle.shape}")

        # Apply projection
        if self.constraint_type == "lower":
            u_proj = np.maximum(u, self.obstacle)
        else:  # upper
            u_proj = np.minimum(u, self.obstacle)

        # Apply regional mask if provided
        if self.region is not None:
            u_proj = np.where(self.region, u_proj, u)

        return u_proj

    def is_feasible(self, u: NDArray, tol: float = 1e-10) -> bool:
        """
        Check if field satisfies constraint.

        Tests whether u ∈ K within tolerance tol.

        Args:
            u: Field to check, shape (Nx, Ny, ...)
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
        # Validate shape
        if u.shape != self.obstacle.shape:
            raise ValueError(f"Field shape {u.shape} doesn't match obstacle shape {self.obstacle.shape}")

        # Check constraint
        if self.constraint_type == "lower":
            feasible = np.all(u >= self.obstacle - tol)
        else:  # upper
            feasible = np.all(u <= self.obstacle + tol)

        # If region mask provided, only check constrained region
        if self.region is not None:
            if self.constraint_type == "lower":
                feasible = np.all((u >= self.obstacle - tol) | ~self.region)
            else:  # upper
                feasible = np.all((u <= self.obstacle + tol) | ~self.region)

        return bool(feasible)

    def get_active_set(self, u: NDArray, tol: float = 1e-10) -> NDArray:
        """
        Identify active constraints (where constraint binds).

        For obstacle constraint:
            - Lower (u ≥ ψ): Active where u ≈ ψ (touching lower bound)
            - Upper (u ≤ ψ): Active where u ≈ ψ (touching upper bound)

        Args:
            u: Field to analyze, shape (Nx, Ny, ...)
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
            For bilateral constraints, use BilateralConstraint which returns
            mask where EITHER lower or upper constraint is active.
        """
        # Validate shape
        if u.shape != self.obstacle.shape:
            raise ValueError(f"Field shape {u.shape} doesn't match obstacle shape {self.obstacle.shape}")

        # Binding means "at the bound OR past it", derived from the SAME inequality `is_feasible`
        # tests, so the two predicates cannot disagree about what "at the bound" means.
        #
        # ~~`np.abs(u - psi) < tol`~~ [FIXED 2026-08-15, #1941] was two-sided while `is_feasible` is
        # one-sided, so a point that VIOLATES the constraint was reported INACTIVE. Measured with
        # `lower = -1` and `u = [-2, -1, 0, 1, 2]`: `is_feasible` False, active set
        # `[False, True, False, False, False]` -- `u[0]` violates by 1.0 and is inactive, while
        # `u[1]`, exactly on the bound, is active. At `tol = 1e-10` a violation of 1e-9 was both
        # infeasible and inactive.
        #
        # That matters beyond the inconsistency: an active-set method iterates on the constraints it
        # believes are binding, and it runs on infeasible iterates by construction -- that is the
        # state before projection. The old form returned an empty active set on exactly the points
        # that needed attention.
        if self.constraint_type == "lower":
            active = u <= self.obstacle + tol
        else:  # upper
            active = u >= self.obstacle - tol

        # Apply regional mask if provided
        if self.region is not None:
            active = active & self.region

        return active

    def __repr__(self) -> str:
        """String representation of constraint."""
        region_str = "everywhere" if self.region is None else f"regional ({self.region.sum()} points)"
        return (
            f"ObstacleConstraint(\n"
            f"  type='{self.constraint_type}',\n"
            f"  obstacle_shape={self.obstacle.shape},\n"
            f"  region={region_str}\n"
            f")"
        )


class BilateralConstraint:
    """
    Bilateral constraint (box constraint): ψ_lower ≤ u ≤ ψ_upper.

    Implements ConstraintProtocol for variational inequalities with two-sided
    bounds, common in control problems with bounded controls and state constraints.

    **Mathematical Context**:
        K = {u : ψ_lower ≤ u ≤ ψ_upper}

        Projection:
            P_K(u) = clip(u, ψ_lower, ψ_upper)
                   = max(ψ_lower, min(u, ψ_upper))

    **MFG Applications**:
        1. **Bounded controls**: α ∈ [α_min, α_max]
        2. **State constraints**: x ∈ [x_min, x_max]
        3. **Probability bounds**: m ∈ [m_min, m_max]

    **Projection Properties**:
        - **Idempotent**: project(project(u)) = project(u)  ✓
        - **Non-expansive**: ‖P_K(u) - P_K(v)‖ ≤ ‖u - v‖  ✓
        - **Feasibility**: is_feasible(project(u)) = True  ✓

    Attributes:
        lower_bound: Lower bound ψ_lower, shape (Nx, Ny, ...)
        upper_bound: Upper bound ψ_upper, shape (Nx, Ny, ...)
        region: Optional mask for regional constraints

    Example:
        >>> import numpy as np
        >>> from mfgarchon.geometry.boundary.constraints import BilateralConstraint
        >>>
        >>> # Bounded control: α ∈ [-1, 1]
        >>> grid_shape = (100, 100)
        >>> lower = -np.ones(grid_shape)
        >>> upper = np.ones(grid_shape)
        >>> constraint = BilateralConstraint(lower, upper)
        >>>
        >>> # Check if control is feasible
        >>> alpha = 1.5 * np.ones(grid_shape)  # Exceeds upper bound
        >>> assert not constraint.is_feasible(alpha)
        >>>
        >>> # Enforce bounds
        >>> alpha_feasible = constraint.project(alpha)
        >>> assert constraint.is_feasible(alpha_feasible)
        >>> assert np.allclose(alpha_feasible, 1.0)  # Clipped to upper bound
    """

    def __init__(
        self,
        lower_bound: NDArray,
        upper_bound: NDArray,
        region: NDArray | None = None,
    ):
        """
        Initialize bilateral constraint.

        Args:
            lower_bound: Lower bound ψ_lower, shape (Nx, Ny, ...)
            upper_bound: Upper bound ψ_upper, shape (Nx, Ny, ...)
            region: Optional boolean mask, same shape as bounds
                - True: Apply constraint in this region
                - False: Free (no constraint)
                - None: Apply constraint everywhere (default)

        Raises:
            ValueError: If bounds have different shapes
            ValueError: If lower_bound > upper_bound anywhere
            ValueError: If region shape doesn't match bounds

        Example:
            >>> # Bounded control α ∈ [-1, 1]
            >>> lower = -np.ones((50, 50))
            >>> upper = np.ones((50, 50))
            >>> constraint = BilateralConstraint(lower, upper)
            >>>
            >>> # State-dependent bounds
            >>> lower = -0.5 * np.ones((50, 50))
            >>> upper = 0.5 + 0.3 * X  # Upper bound varies with position
            >>> constraint = BilateralConstraint(lower, upper)
        """
        if lower_bound.shape != upper_bound.shape:
            raise ValueError(
                f"lower_bound shape {lower_bound.shape} doesn't match upper_bound shape {upper_bound.shape}"
            )

        if np.any(lower_bound > upper_bound):
            raise ValueError("lower_bound must be ≤ upper_bound everywhere")

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.region = region

        # Validate region shape
        if region is not None:
            if region.shape != lower_bound.shape:
                raise ValueError(f"region shape {region.shape} doesn't match bounds shape {lower_bound.shape}")

    def project(self, u: NDArray) -> NDArray:
        """
        Project field onto constraint set K.

        Computes P_K(u) = clip(u, ψ_lower, ψ_upper).

        Args:
            u: Field to project, shape (Nx, Ny, ...)

        Returns:
            Projected field P_K(u), same shape as u

        Example:
            >>> u = np.array([0.5, -1.5, 2.0])
            >>> lower = np.array([-1.0, -1.0, -1.0])
            >>> upper = np.array([1.0, 1.0, 1.0])
            >>> constraint = BilateralConstraint(lower, upper)
            >>> u_proj = constraint.project(u)
            >>> # u_proj = [0.5, -1.0, 1.0]  (clips to [-1, 1])
        """
        if u.shape != self.lower_bound.shape:
            raise ValueError(f"Field shape {u.shape} doesn't match bounds shape {self.lower_bound.shape}")

        # Clip to bounds
        u_proj = np.clip(u, self.lower_bound, self.upper_bound)

        # Apply regional mask if provided
        if self.region is not None:
            u_proj = np.where(self.region, u_proj, u)

        return u_proj

    def is_feasible(self, u: NDArray, tol: float = 1e-10) -> bool:
        """
        Check if field satisfies bilateral constraint.

        Tests whether ψ_lower - tol ≤ u ≤ ψ_upper + tol.

        Args:
            u: Field to check, shape (Nx, Ny, ...)
            tol: Tolerance for constraint violation (default 1e-10)

        Returns:
            True if u ∈ K (within tol), False otherwise
        """
        if u.shape != self.lower_bound.shape:
            raise ValueError(f"Field shape {u.shape} doesn't match bounds shape {self.lower_bound.shape}")

        # Check both bounds
        feasible = np.all((u >= self.lower_bound - tol) & (u <= self.upper_bound + tol))

        # If region mask provided, only check constrained region
        if self.region is not None:
            feasible = np.all(((u >= self.lower_bound - tol) & (u <= self.upper_bound + tol)) | ~self.region)

        return bool(feasible)

    def get_active_set(self, u: NDArray, tol: float = 1e-10) -> NDArray:
        """
        Identify active constraints (where either bound binds).

        Returns mask where EITHER lower or upper constraint is active.

        Args:
            u: Field to analyze, shape (Nx, Ny, ...)
            tol: Tolerance for detecting binding (default 1e-10)

        Returns:
            Boolean mask, same shape as u
                - True: either bound is active (binding)
                - False: both bounds inactive (has slack)

        Example:
            >>> u = np.array([0.0, -1.0, 1.0])
            >>> lower = np.array([-1.0, -1.0, -1.0])
            >>> upper = np.array([1.0, 1.0, 1.0])
            >>> constraint = BilateralConstraint(lower, upper)
            >>> active = constraint.get_active_set(u, tol=1e-8)
            >>> # active = [False, True, True]  (last two at bounds)
        """
        if u.shape != self.lower_bound.shape:
            raise ValueError(f"Field shape {u.shape} doesn't match bounds shape {self.lower_bound.shape}")

        # Active if AT or PAST either bound, derived from the same inequalities `is_feasible` tests.
        # A third copy of the two-sided form lived here (`|u - bound| < tol`) and had the same
        # defect as `ObstacleConstraint.get_active_set`: a point below the lower bound or above the
        # upper one was reported inactive. Found by the test written for the obstacle case. #1941
        active_lower = u <= self.lower_bound + tol
        active_upper = u >= self.upper_bound - tol
        active = active_lower | active_upper

        # Apply regional mask if provided
        if self.region is not None:
            active = active & self.region

        return active

    def __repr__(self) -> str:
        """String representation of constraint."""
        region_str = "everywhere" if self.region is None else f"regional ({self.region.sum()} points)"
        return f"BilateralConstraint(\n  bounds_shape={self.lower_bound.shape},\n  region={region_str}\n)"


if __name__ == "__main__":
    """Smoke test for constraint implementations."""
    import numpy as np

    print("Testing constraint implementations...")

    # ========================================
    # Test 1: ObstacleConstraint - Lower
    # ========================================
    print("\n[Test 1: ObstacleConstraint - Lower (u ≥ ψ)]")
    psi = np.array([0.5, 1.0, 0.0])
    constraint = ObstacleConstraint(psi, constraint_type="lower")

    print(f"  Obstacle: {psi}")
    print(f"  Constraint: {constraint}")

    # Test projection
    u = np.array([1.0, 0.5, -0.5])
    u_proj = constraint.project(u)
    expected = np.array([1.0, 1.0, 0.0])  # Enforces u >= psi

    print(f"  Input: {u}")
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
    assert not constraint.is_feasible(u), "Infeasible u should fail"
    assert constraint.is_feasible(u_proj), "Projected u should be feasible"
    print("  ✓ Feasibility check works")

    # Test active set
    active = constraint.get_active_set(u_proj, tol=1e-8)
    expected_active = np.array([False, True, True])  # Last two at obstacle

    print(f"  Active set: {active}")
    print(f"  Expected: {expected_active}")
    assert np.array_equal(active, expected_active), "Active set incorrect"
    print("  ✓ Active set detection works")

    # ========================================
    # Test 2: ObstacleConstraint - Upper
    # ========================================
    print("\n[Test 2: ObstacleConstraint - Upper (u ≤ ψ)]")
    m_max = np.array([0.5, 0.5, 0.5])
    constraint_upper = ObstacleConstraint(m_max, constraint_type="upper")

    m = np.array([0.3, 0.7, 0.4])
    m_proj = constraint_upper.project(m)
    expected_m = np.array([0.3, 0.5, 0.4])  # Clips to capacity

    print(f"  Capacity limit: {m_max}")
    print(f"  Input density: {m}")
    print(f"  Projected: {m_proj}")
    print(f"  Expected: {expected_m}")
    assert np.allclose(m_proj, expected_m), "Upper projection failed"
    print("  ✓ Upper projection works")

    # ========================================
    # Test 3: ObstacleConstraint - Regional
    # ========================================
    print("\n[Test 3: ObstacleConstraint - Regional]")
    psi_2d = np.zeros((5, 5))
    region = np.zeros((5, 5), dtype=bool)
    region[2, 2] = True  # Only constrain center point

    constraint_regional = ObstacleConstraint(psi_2d, constraint_type="lower", region=region)

    u_2d = -np.ones((5, 5))  # All negative
    u_proj_2d = constraint_regional.project(u_2d)

    # Only center should be projected to 0, rest unchanged
    expected_2d = -np.ones((5, 5))
    expected_2d[2, 2] = 0.0

    print(f"  Region: {region.sum()} constrained points")
    print(f"  Projected center: {u_proj_2d[2, 2]:.2f} (expected 0.0)")
    print(f"  Projected corner: {u_proj_2d[0, 0]:.2f} (expected -1.0)")
    assert np.allclose(u_proj_2d, expected_2d), "Regional constraint failed"
    print("  ✓ Regional constraint works")

    # ========================================
    # Test 4: BilateralConstraint
    # ========================================
    print("\n[Test 4: BilateralConstraint (ψ_lower ≤ u ≤ ψ_upper)]")
    lower = np.array([-1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0])
    constraint_bilateral = BilateralConstraint(lower, upper)

    print(f"  Bounds: [{lower[0]:.1f}, {upper[0]:.1f}]")

    # Test projection
    u = np.array([0.5, -1.5, 2.0])
    u_proj = constraint_bilateral.project(u)
    expected = np.array([0.5, -1.0, 1.0])  # Clips to [-1, 1]

    print(f"  Input: {u}")
    print(f"  Projected: {u_proj}")
    print(f"  Expected: {expected}")
    assert np.allclose(u_proj, expected), "Bilateral projection failed"
    print("  ✓ Bilateral projection works")

    # Test feasibility
    assert not constraint_bilateral.is_feasible(u), "Out-of-bounds u should fail"
    assert constraint_bilateral.is_feasible(u_proj), "Projected u should be feasible"
    print("  ✓ Bilateral feasibility works")

    # Test active set (both bounds)
    active = constraint_bilateral.get_active_set(u_proj, tol=1e-8)
    expected_active = np.array([False, True, True])  # Last two at bounds

    print(f"  Active set: {active}")
    print(f"  Expected: {expected_active}")
    assert np.array_equal(active, expected_active), "Bilateral active set incorrect"
    print("  ✓ Bilateral active set works")

    # ========================================
    # Test 5: Protocol Compliance
    # ========================================
    print("\n[Test 5: Protocol Compliance]")
    from mfgarchon.geometry.boundary.constraint_protocol import ConstraintProtocol

    # Check ObstacleConstraint
    assert isinstance(constraint, ConstraintProtocol), "ObstacleConstraint should implement protocol"
    print("  ✓ ObstacleConstraint implements ConstraintProtocol")

    # Check BilateralConstraint
    assert isinstance(constraint_bilateral, ConstraintProtocol), "BilateralConstraint should implement protocol"
    print("  ✓ BilateralConstraint implements ConstraintProtocol")

    print("\n✅ All constraint implementation tests passed!")
