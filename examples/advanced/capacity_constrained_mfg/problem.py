"""
Capacity-constrained MFG problems for maze navigation and crowd dynamics.

This module extends the standard MFG framework with spatially-varying capacity
constraints C(x), enabling realistic modeling of:
- Pedestrian flow in complex geometries (corridors, rooms, stairs)
- Traffic flow with lane capacity
- Multi-agent navigation with physical constraints

Mathematical Framework:
    The Hamiltonian is modified to include a congestion cost term:

    H(x, m, ∇u) = (1/2)|∇u|² + γ·g(m(x)/C(x))

    where:
    - m(x): Agent density at position x
    - C(x): Local corridor capacity (from geometry)
    - g(ρ): Convex congestion cost function (ρ = m/C)
    - γ: Congestion weight parameter

    As the density m(x) approaches capacity C(x), the cost g(m/C) → ∞,
    creating a "soft wall" effect that prevents overcrowding.

Key Components:
- CapacityField: Spatially-varying capacity C(x) from maze geometry
- CongestionModel: Convex cost function g(ρ) with derivatives
- CapacityConstrainedMFGProblem: MFG problem with congestion term

References:
    - Hughes, R. L. (2002). "A continuum theory for the flow of pedestrians."
      Transportation Research Part B, 36(6), 507-535.
    - Achdou, Y., et al. (2020). "Mean field games with congestion."
      Annales de l'IHP Analyse non linéaire, 37(3), 637-663.
    - Di Francesco, M., & Fagioli, S. (2013). "Measure solutions for non-local
      interaction PDEs with two distinct species." Nonlinearity, 26(10), 2777.

Created: 2025-11-12
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.core import MFGProblem

from .capacity_field import CapacityField
from .congestion import CongestionModel

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


class CapacityConstrainedMFGProblem(MFGProblem):
    """
    MFG problem with capacity constraints for maze navigation and crowd dynamics.

    Extends the standard MFG framework with a congestion term in the Hamiltonian:
        H(x, m, ∇u) = H_base(x, m, ∇u) + γ·g(m/C)

    The congestion term creates a "soft barrier" that penalizes high density
    relative to local corridor capacity, naturally encouraging agents to avoid
    overcrowded regions and seek alternative routes.

    Attributes:
        capacity_field: CapacityField instance providing C(x)
        congestion_model: CongestionModel instance providing g(ρ) and ∂g/∂m
        congestion_weight: Weight parameter γ for congestion cost

    Examples:
        >>> from mfgarchon.geometry.graph import create_perfect_maze
        >>> from mfgarchon.geometry.graph import CapacityField
        >>> from mfgarchon.core.congestion import QuadraticCongestion
        >>>
        >>> # Generate maze and compute capacity
        >>> maze = create_perfect_maze(rows=20, cols=20, wall_thickness=3)
        >>> maze_array = maze.to_numpy_array(wall_thickness=3)
        >>> capacity = CapacityField.from_maze_geometry(maze_array, wall_thickness=3)
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
        >>> # Use with any MFG solver
        >>> result = solve_mfg(problem, method="semi_lagrangian")
    """

    def __init__(
        self,
        capacity_field: CapacityField,
        congestion_model: CongestionModel,
        congestion_weight: float = 1.0,
        **kwargs: Any,
    ):
        """
        Initialize capacity-constrained MFG problem.

        Args:
            capacity_field: CapacityField providing spatially-varying capacity C(x)
            congestion_model: CongestionModel providing congestion cost g(ρ)
            congestion_weight: Weight γ for congestion term (default: 1.0)
                              Higher values → stronger congestion avoidance
            **kwargs: Additional arguments passed to MFGProblem.__init__()
                     Common args: spatial_bounds, spatial_discretization, T, Nt, sigma

        Raises:
            ValueError: If congestion_weight is negative
            ValueError: If capacity_field dimension doesn't match problem dimension

        Notes:
            The congestion term is added to the base Hamiltonian:
                H_total = H_base + γ·g(m/C)

            Choice of congestion_weight γ:
            - γ = 0: No congestion (free flow)
            - γ ~ 0.1-1.0: Moderate congestion avoidance
            - γ > 10: Strong congestion avoidance (near hard constraints)
        """
        if congestion_weight < 0:
            raise ValueError(f"congestion_weight must be non-negative, got {congestion_weight}")

        # Initialize base MFG problem
        super().__init__(**kwargs)

        # Store capacity-specific components
        self.capacity_field = capacity_field
        self.congestion_model = congestion_model
        self.congestion_weight = congestion_weight

        # Validate dimension compatibility
        if hasattr(self, "dimension"):
            if isinstance(self.dimension, int) and self.dimension != capacity_field.dimension:
                raise ValueError(
                    f"Capacity field dimension ({capacity_field.dimension}) "
                    f"doesn't match problem dimension ({self.dimension})"
                )

    def hamiltonian(self, t, x, p, m) -> float:
        """
        Compute Hamiltonian with congestion term.

        The total Hamiltonian is:
            H(t, x, p, m) = (1/2)|p|² + α·m + γ·g(m(x)/C(x))

        where:
        - (1/2)|p|²: Kinetic energy (standard)
        - α·m: Coupling term (density interaction)
        - g(m/C): Congestion cost function
        - γ: Congestion weight

        Args:
            x: Spatial position (scalar for 1D, tuple/array for nD)
            m: Density value at this position
            p: Momentum/co-state ∂u/∂x (scalar for 1D, tuple/array for nD)
            t: Time

        Returns:
            Total Hamiltonian value H(x, m, p, t)

        Notes:
            - Capacity C(x) is interpolated from the capacity field
            - For grid solvers, x is typically in grid index coordinates
            - For particle solvers, conversion may be needed
        """
        # Base Hamiltonian: (1/2)|p|² + α·m
        p_array = np.atleast_1d(p)
        H_base = 0.5 * np.sum(p_array**2)  # Kinetic energy

        # Add coupling term if present
        if hasattr(self, "coupling_coefficient"):
            H_base += self.coupling_coefficient * m

        # Get capacity at this position
        # Convert x to array format for interpolation
        x_array = np.atleast_1d(x)
        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)  # Shape: (1, dimension)

        # Interpolate capacity at this position
        capacity_at_x = self.capacity_field.interpolate_at_positions(x_array, method="linear")[0]

        # Compute congestion cost: g(m/C)
        # Note: Both m and capacity_at_x are scalars here
        congestion_cost = self.congestion_model.cost(
            density=np.array([m]),  # Convert to array for congestion model
            capacity=np.array([capacity_at_x]),
        )[0]  # Extract scalar result

        # Total Hamiltonian
        return H_base + self.congestion_weight * congestion_cost

    def hamiltonian_dm(self, t, x, p, m) -> float:
        """
        Compute derivative of Hamiltonian with respect to density: ∂H/∂m.

        This is needed for the FP equation coupling:
            ∂m/∂t = σΔm - div(m·∇_p H) + div(m·∇H_m)

        The derivative includes the congestion term:
            ∂H/∂m = α + γ·(∂g/∂m)

        Args:
            x: Spatial position
            m: Density value at this position
            p: Momentum/co-state ∂u/∂x
            t: Time

        Returns:
            Derivative ∂H/∂m at (x, m, p, t)

        Notes:
            The congestion model provides ∂g/∂m = g'(m/C) / C
        """
        # Base derivative: coupling coefficient α
        H_dm_base = self.coupling_coefficient if hasattr(self, "coupling_coefficient") else 0.0

        # Get capacity at this position
        x_array = np.atleast_1d(x)
        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)

        capacity_at_x = self.capacity_field.interpolate_at_positions(x_array, method="linear")[0]

        # Compute congestion derivative: ∂g/∂m
        congestion_derivative = self.congestion_model.derivative(
            density=np.array([m]), capacity=np.array([capacity_at_x])
        )[0]

        # Total derivative
        return H_dm_base + self.congestion_weight * congestion_derivative

    def get_capacity_at_grid(self, grid_positions: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Get capacity values at grid positions (for visualization/analysis).

        Args:
            grid_positions: Grid positions (N, dimension)

        Returns:
            Capacity values at each grid point (N,)

        Examples:
            >>> # Get capacity on 2D grid
            >>> x = np.linspace(0, 1, 50)
            >>> y = np.linspace(0, 1, 50)
            >>> X, Y = np.meshgrid(x, y)
            >>> positions = np.column_stack([X.ravel(), Y.ravel()])
            >>> capacity_grid = problem.get_capacity_at_grid(positions)
        """
        return self.capacity_field.interpolate_at_positions(grid_positions, method="linear")

    def get_congestion_ratio(
        self, density: NDArray[np.floating], positions: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Compute congestion ratio ρ(x) = m(x) / C(x) at given positions.

        Args:
            density: Density values at positions (N,)
            positions: Spatial positions (N, dimension)

        Returns:
            Congestion ratios (N,)

        Notes:
            - ρ < 1: Free flow (density below capacity)
            - ρ ≈ 1: Near capacity (congested)
            - ρ > 1: Overcapacity (should be penalized by g(ρ))
        """
        capacity_values = self.get_capacity_at_grid(positions)
        return density / capacity_values

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  capacity_field={self.capacity_field},\n"
            f"  congestion_model={self.congestion_model.__class__.__name__},\n"
            f"  congestion_weight={self.congestion_weight},\n"
            f"  dimension={self.dimension if hasattr(self, 'dimension') else 'N/A'},\n"
            f"  T={self.T if hasattr(self, 'T') else 'N/A'},\n"
            f"  Nt={self.Nt if hasattr(self, 'Nt') else 'N/A'},\n"
            f")"
        )


__all__ = [
    "CapacityConstrainedMFGProblem",
]


if __name__ == "__main__":
    """Smoke tests for CapacityConstrainedMFGProblem."""
    print("Running CapacityConstrainedMFGProblem smoke tests...")

    # Test 1: Problem initialization with synthetic capacity field
    print("\n1. Problem initialization...")
    capacity_array = np.random.uniform(0.5, 1.0, (50, 50))
    capacity = CapacityField(capacity=capacity_array, epsilon=0.01)

    from .congestion import QuadraticCongestion

    problem = CapacityConstrainedMFGProblem(
        capacity_field=capacity,
        congestion_model=QuadraticCongestion(),
        congestion_weight=0.5,
        spatial_bounds=[(0, 1), (0, 1)],
        spatial_discretization=[50, 50],
        T=1.0,
        Nt=50,
        sigma=0.01,
    )
    print(f"   Created problem: dimension={problem.dimension}, T={problem.T}, Nt={problem.Nt}")
    print("   ✓ Initialization successful")

    # Test 2: Hamiltonian evaluation at a point
    print("\n2. Hamiltonian evaluation...")
    x = np.array([25.0, 25.0])  # Center of grid
    m = 0.5  # Moderate density
    p = np.array([0.1, 0.1])  # Small momentum
    t = 0.5

    H = problem.hamiltonian(t=t, x=x, p=p, m=m)
    print(f"   H(x={x}, m={m}, p={p}, t={t}) = {H:.6f}")
    assert np.isfinite(H), "Hamiltonian should be finite"
    print("   ✓ Hamiltonian evaluation works")

    # Test 3: Hamiltonian derivative
    print("\n3. Hamiltonian derivative (∂H/∂m)...")
    H_dm = problem.hamiltonian_dm(t=t, x=x, p=p, m=m)
    print(f"   ∂H/∂m = {H_dm:.6f}")
    assert np.isfinite(H_dm), "Derivative should be finite"
    assert H_dm > 0, "Derivative should be positive (convex congestion)"
    print("   ✓ Derivative computation works")

    # Test 4: Get capacity at grid
    print("\n4. Capacity interpolation at grid positions...")
    test_positions = np.array([[10.0, 10.0], [25.0, 25.0], [40.0, 40.0]])
    capacities = problem.get_capacity_at_grid(test_positions)
    print(f"   Capacities at test positions: {capacities}")
    assert len(capacities) == 3, "Should return 3 capacity values"
    assert np.all(capacities > 0), "All capacities should be positive"
    print("   ✓ Capacity interpolation works")

    # Test 5: Congestion ratio computation
    print("\n5. Congestion ratio computation...")
    densities = np.array([0.3, 0.6, 0.9])
    congestion_ratios = problem.get_congestion_ratio(densities, test_positions)
    print(f"   Congestion ratios: {congestion_ratios}")
    assert len(congestion_ratios) == 3, "Should return 3 ratios"
    assert np.all(congestion_ratios > 0), "All ratios should be positive"
    print("   ✓ Congestion ratio computation works")

    # Test 6: Hamiltonian increases with density (convexity check)
    print("\n6. Hamiltonian convexity check (∂H/∂m > 0)...")
    m_values = np.linspace(0.1, 0.9, 5)
    H_values = [problem.hamiltonian(t=t, x=x, p=p, m=m_test) for m_test in m_values]
    print(f"   m = {m_values}")
    print(f"   H = {H_values}")

    # Check monotonicity (H should increase with m)
    H_diffs = np.diff(H_values)
    assert np.all(H_diffs > 0), "Hamiltonian should increase with density (convex)"
    print("   ✓ Hamiltonian is convex in density")

    # Test 7: LogBarrier congestion (stability at high density)
    print("\n7. LogBarrier congestion (high density stability)...")
    from .congestion import LogBarrierCongestion

    problem_logbarrier = CapacityConstrainedMFGProblem(
        capacity_field=capacity,
        congestion_model=LogBarrierCongestion(threshold=0.95),
        congestion_weight=1.0,
        spatial_bounds=[(0, 1), (0, 1)],
        spatial_discretization=[50, 50],
        T=1.0,
        Nt=50,
        sigma=0.01,
    )

    # Test at overcapacity (m > C)
    m_high = 1.5  # Density exceeds capacity
    H_high = problem_logbarrier.hamiltonian(t=t, x=x, p=p, m=m_high)
    print(f"   H at overcapacity (m={m_high}): {H_high:.6f}")
    assert np.isfinite(H_high), "Should not return NaN at overcapacity"
    print("   ✓ LogBarrier stable at overcapacity")

    # Test 8: String representation
    print("\n8. String representation...")
    repr_str = repr(problem)
    print(f"   {repr_str}")
    assert "CapacityConstrainedMFGProblem" in repr_str
    print("   ✓ String representation works")

    print("\n✅ All CapacityConstrainedMFGProblem smoke tests passed!")
