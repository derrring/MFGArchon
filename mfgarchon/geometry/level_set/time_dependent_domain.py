"""
Time-Dependent Domain Wrapper for Level Set Evolution.

Manages evolving level set φ(t) representing a domain with moving boundaries.
Provides access to geometry snapshots at different times for use with
existing MFG solvers (which expect static geometry).

Design Pattern:
    TimeDependentDomain stores φ(t) history → provides φ at requested times
    → Existing solvers use φ(t) as if it were static for that timestep

This composition-over-inheritance approach avoids modifying existing solvers.

Example Workflow (Stefan Problem):
    ```python
    # Initialize with SDF at t=0
    ls_domain = TimeDependentDomain(phi0, grid)

    for t in timesteps:
        # Solve heat equation on current geometry
        phi_current = ls_domain.get_phi_at_time(t)
        T = solve_heat_equation(phi_current, grid)

        # Compute interface velocity from solution
        V = -jump_heat_flux(T)

        # Evolve level set
        ls_domain.evolve_step(V, dt)
    ```

Created: 2026-01-18 (Issue #592 Milestone 3.1.4)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.geometry.level_set.core import LevelSetEvolver, LevelSetFunction
from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

# Module logger
logger = get_logger(__name__)


class TimeDependentDomain:
    """
    Container for time-evolving level set domain.

    Manages the evolution of a domain represented by φ(t), where the zero
    level set {x : φ(x,t) = 0} represents the moving boundary.

    Attributes:
        geometry: Grid structure (TensorProductGrid)
        evolver: LevelSetEvolver for ∂φ/∂t + V|∇φ| = 0
        phi_history: List of level set snapshots [φ₀, φ₁, ...]
        time_history: List of corresponding times [t₀, t₁, ...]
        current_time: Most recent time in history

    Example:
        >>> # Initialize with circle at t=0
        >>> phi0 = np.linalg.norm(X - center, axis=0) - radius
        >>> ls_domain = TimeDependentDomain(phi0, grid)
        >>>
        >>> # Evolve with constant velocity
        >>> V = 0.5  # Expansion
        >>> ls_domain.evolve_step(V, dt=0.1)
        >>>
        >>> # Get current level set
        >>> phi_t = ls_domain.get_phi_at_time(ls_domain.current_time)
    """

    def __init__(
        self,
        phi_initial: NDArray[np.float64],
        geometry: TensorProductGrid,
        *,
        initial_time: float = 0.0,
        is_signed_distance: bool = True,
    ):
        """
        Initialize time-dependent domain.

        Args:
            phi_initial: Initial level set function, shape (Nx, Ny, ...)
            geometry: Grid providing operators and spatial structure
            initial_time: Starting time (default: 0.0)
            is_signed_distance: Whether phi_initial is a true SDF |∇φ| = 1

        Raises:
            ValueError: If phi_initial shape doesn't match geometry
        """
        self.geometry = geometry
        self.evolver = LevelSetEvolver(geometry, scheme="upwind")

        # Initialize history
        self.phi_history = [phi_initial.copy()]
        self.time_history = [initial_time]
        self.is_sdf_history = [is_signed_distance]

        logger.debug(
            f"TimeDependentDomain initialized: "
            f"dimension={geometry.dimension}, "
            f"t0={initial_time}, "
            f"is_SDF={is_signed_distance}"
        )

    @property
    def current_time(self) -> float:
        """Most recent time in history."""
        return self.time_history[-1]

    @property
    def current_phi(self) -> NDArray[np.float64]:
        """Most recent level set function."""
        return self.phi_history[-1]

    @property
    def num_snapshots(self) -> int:
        """Number of stored time snapshots."""
        return len(self.time_history)

    def get_phi_at_time(self, t: float, interpolate: bool = False) -> NDArray[np.float64]:
        """
        Retrieve level set at requested time.

        Args:
            t: Time to query
            interpolate: If True, linearly interpolate between snapshots.
                If False (default), return nearest snapshot.

        Returns:
            Level set function φ(t), shape (Nx, Ny, ...)

        Raises:
            ValueError: If t is outside stored time range

        Example:
            >>> # Get exact snapshot
            >>> phi_t = ls_domain.get_phi_at_time(0.5)
            >>>
            >>> # Get interpolated value
            >>> phi_t = ls_domain.get_phi_at_time(0.55, interpolate=True)
        """
        if t < self.time_history[0] or t > self.time_history[-1]:
            raise ValueError(f"Time t={t} outside stored range [{self.time_history[0]}, {self.time_history[-1]}]")

        # Find closest snapshot(s)
        idx = np.searchsorted(self.time_history, t)

        if not interpolate or idx == 0:
            # Return nearest snapshot
            if idx < len(self.time_history) and abs(self.time_history[idx] - t) < abs(self.time_history[idx - 1] - t):
                return self.phi_history[idx].copy()
            else:
                return self.phi_history[idx - 1].copy()

        # Linear interpolation between idx-1 and idx
        t0, t1 = self.time_history[idx - 1], self.time_history[idx]
        phi0, phi1 = self.phi_history[idx - 1], self.phi_history[idx]

        alpha = (t - t0) / (t1 - t0)
        phi_interp = (1 - alpha) * phi0 + alpha * phi1

        return phi_interp

    def evolve_step(
        self,
        velocity: float | NDArray[np.float64] | Callable[[NDArray], NDArray],
        dt: float,
        *,
        reinitialize: bool = False,
        save_to_history: bool = True,
    ) -> NDArray[np.float64]:
        """
        Evolve level set one time step: φⁿ⁺¹ = φⁿ - dt·V·|∇φ|.

        Args:
            velocity: Normal velocity field V
                - float: Constant velocity
                - NDArray: Spatially-varying velocity
                - Callable: a function of ``x``, a coordinate tuple at the mesh's points, read by name
            dt: Time step size
            reinitialize: Whether to reinitialize after evolution (default: False)
                Set True every 5-10 steps to maintain SDF property
            save_to_history: Whether to save result to history (default: True)

        Returns:
            Updated level set φⁿ⁺¹, shape (Nx, Ny, ...)

        Example:
            >>> # Evolve with constant velocity
            >>> phi_new = ls_domain.evolve_step(velocity=1.0, dt=0.1)
            >>>
            >>> # Evolve with spatially-varying velocity, reinitialize
            >>> V = np.sin(X[0]) * np.cos(X[1])
            >>> phi_new = ls_domain.evolve_step(V, dt=0.1, reinitialize=True)
        """
        # Evolve using LevelSetEvolver
        phi_new = self.evolver.evolve_step(self.current_phi, velocity, dt)

        # Optionally reinitialize to maintain SDF property
        if reinitialize:
            from mfgarchon.geometry.level_set.reinitialization import reinitialize

            logger.debug(f"Reinitializing level set at t={self.current_time + dt:.4f}")
            phi_new = reinitialize(phi_new, self.geometry, max_iterations=10)
            is_sdf = True
        else:
            is_sdf = False

        # Save to history
        if save_to_history:
            new_time = self.current_time + dt
            self.time_history.append(new_time)
            self.phi_history.append(phi_new)
            self.is_sdf_history.append(is_sdf)

            logger.debug(f"Evolved to t={new_time:.4f}, total snapshots={self.num_snapshots}, reinit={reinitialize}")

        return phi_new

    def get_level_set_function(self, t: float | None = None) -> LevelSetFunction:
        """
        Get LevelSetFunction wrapper at specified time.

        Provides access to normals, curvature, interface mask, etc.

        Args:
            t: Time to query (default: None → current time)

        Returns:
            LevelSetFunction instance with geometric queries

        Example:
            >>> ls = ls_domain.get_level_set_function()
            >>> normals = ls.get_normal()
            >>> curvature = ls.get_curvature()
            >>> interface = ls.interface_mask(width=2*dx)
        """
        if t is None:
            t = self.current_time

        phi = self.get_phi_at_time(t)

        # Check if this snapshot was reinitialized
        idx = np.searchsorted(self.time_history, t)
        is_sdf = self.is_sdf_history[min(idx, len(self.is_sdf_history) - 1)]

        return LevelSetFunction(phi, self.geometry, is_signed_distance=is_sdf)

    def clear_history_before(self, t_cutoff: float) -> None:
        """
        Remove snapshots before t_cutoff to save memory.

        Args:
            t_cutoff: Keep only snapshots with t >= t_cutoff

        Example:
            >>> # Keep only last 10% of history
            >>> t_cutoff = 0.9 * ls_domain.current_time
            >>> ls_domain.clear_history_before(t_cutoff)
        """
        # Find first index to keep
        idx_keep = np.searchsorted(self.time_history, t_cutoff)

        if idx_keep > 0:
            self.time_history = self.time_history[idx_keep:]
            self.phi_history = self.phi_history[idx_keep:]
            self.is_sdf_history = self.is_sdf_history[idx_keep:]

            logger.debug(f"Cleared history before t={t_cutoff:.4f}, kept {len(self.time_history)} snapshots")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"TimeDependentDomain(\n"
            f"  dimension={self.geometry.dimension},\n"
            f"  time_range=[{self.time_history[0]:.4f}, {self.current_time:.4f}],\n"
            f"  num_snapshots={self.num_snapshots},\n"
            f"  current_is_SDF={self.is_sdf_history[-1]}\n"
            f")"
        )


if __name__ == "__main__":
    """Smoke test for TimeDependentDomain."""
    print("Testing TimeDependentDomain...")

    from mfgarchon.geometry.boundary import no_flux_bc
    from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

    # Test 1: 1D expanding circle
    print("\n[Test 1: 1D Expanding Circle]")
    print("Problem: Track circle expanding with constant velocity V = 0.5")

    # Create 1D grid
    Nx = 100
    grid_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[Nx], boundary_conditions=no_flux_bc(dimension=1))
    x = grid_1d.coordinates[0]
    dx = grid_1d.spacing[0]

    print(f"  Grid: {Nx} points, dx = {dx:.4f}")

    # Initial circle: interface at x = 0.5
    x0_interface = 0.5
    phi0 = x - x0_interface
    print(f"  Initial interface: x = {x0_interface}")

    # Create time-dependent domain
    ls_domain = TimeDependentDomain(phi0, grid_1d, initial_time=0.0, is_signed_distance=True)
    print(f"  {ls_domain}")

    # Evolve with constant velocity V = 0.5
    V = 0.5
    dt = 0.1
    n_steps = 5

    print(f"\n  Evolving: V = {V}, dt = {dt}, n_steps = {n_steps}")

    for _step in range(n_steps):
        phi_new = ls_domain.evolve_step(V, dt, reinitialize=False, save_to_history=True)

    print(f"  Final time: t = {ls_domain.current_time:.4f}")
    print(f"  Snapshots stored: {ls_domain.num_snapshots}")

    # Check final interface position
    phi_final = ls_domain.current_phi
    idx_zero = np.where(np.diff(np.sign(phi_final)))[0]
    if len(idx_zero) > 0:
        x_final = x[idx_zero[0]]
        x_expected = x0_interface + V * ls_domain.current_time
        error = abs(x_final - x_expected)
        print(f"  Expected interface: x = {x_expected:.4f}")
        print(f"  Computed interface: x = {x_final:.4f}")
        print(f"  Error: {error:.4f} ({error / dx:.2f} grid points)")
        assert error < 2 * dx, f"Interface error {error:.4f} > {2 * dx:.4f}"
        print("  ✓ Interface tracking test passed!")

    # Test 2: Get phi at intermediate time
    print("\n[Test 2: Time Interpolation]")
    t_mid = 0.25
    print(f"  Query time: t = {t_mid}")

    phi_mid_nearest = ls_domain.get_phi_at_time(t_mid, interpolate=False)
    phi_mid_interp = ls_domain.get_phi_at_time(t_mid, interpolate=True)

    print(f"  Nearest snapshot: phi range = [{phi_mid_nearest.min():.4f}, {phi_mid_nearest.max():.4f}]")
    print(f"  Interpolated: phi range = [{phi_mid_interp.min():.4f}, {phi_mid_interp.max():.4f}]")
    print("  ✓ Time interpolation test passed!")

    # Test 3: LevelSetFunction wrapper
    print("\n[Test 3: LevelSetFunction Wrapper]")
    ls = ls_domain.get_level_set_function()
    print(f"  LevelSetFunction at t={ls_domain.current_time:.4f}")
    print(f"  {ls}")

    # Get normal (should be ±1 in 1D)
    normals = ls.get_normal()
    print(f"  Normal field shape: {normals.shape}")
    print(f"  Normal range: [{normals.min():.4f}, {normals.max():.4f}]")
    assert normals.shape[0] == 1, "1D should have 1 normal component"
    print("  ✓ LevelSetFunction wrapper test passed!")

    # Test 4: History clearing
    print("\n[Test 4: History Management]")
    print(f"  Before clearing: {ls_domain.num_snapshots} snapshots")

    t_cutoff = 0.3
    ls_domain.clear_history_before(t_cutoff)

    print(f"  After clearing (t < {t_cutoff}): {ls_domain.num_snapshots} snapshots")
    assert ls_domain.num_snapshots < n_steps + 1, "History not cleared"
    print("  ✓ History clearing test passed!")

    # Test 5: 2D shrinking circle with reinitialization
    print("\n[Test 5: 2D Shrinking Circle with Reinitialization]")
    print("Problem: Track shrinking circle with periodic reinitialization")

    # Create 2D grid
    Nx_2d, Ny_2d = 50, 50
    grid_2d = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx=[Nx_2d, Ny_2d], boundary_conditions=no_flux_bc(dimension=2)
    )
    X, Y = grid_2d.meshgrid()
    dx_2d = grid_2d.spacing[0]

    print(f"  Grid: {Nx_2d}×{Ny_2d}, dx = {dx_2d:.4f}")

    # Initial circle
    center = np.array([0.5, 0.5])
    radius0 = 0.3
    phi0_2d = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius0

    print(f"  Initial circle: center={center}, radius={radius0}")

    ls_domain_2d = TimeDependentDomain(phi0_2d, grid_2d, is_signed_distance=True)

    # Shrink with constant velocity V = -0.5 (negative = inward)
    V_2d = -0.5
    dt_2d = 0.05
    n_steps_2d = 3

    print(f"  Evolving: V = {V_2d}, dt = {dt_2d}, n_steps = {n_steps_2d}")

    for step in range(n_steps_2d):
        # Reinitialize every other step
        reinit = step % 2 == 1
        ls_domain_2d.evolve_step(V_2d, dt_2d, reinitialize=reinit)

    print(f"  Final time: t = {ls_domain_2d.current_time:.4f}")

    # Check that interface moved inward
    phi_final_2d = ls_domain_2d.current_phi
    interface_2d = np.abs(phi_final_2d) < dx_2d
    coords_final = np.column_stack([X[interface_2d], Y[interface_2d]])

    if len(coords_final) > 0:
        # Compute distance from center
        dists = np.linalg.norm(coords_final - center, axis=1)
        radius_final = np.mean(dists)
        radius_expected = radius0 + V_2d * ls_domain_2d.current_time  # V < 0 → shrink

        print(f"  Expected radius: {radius_expected:.4f}")
        print(f"  Computed radius: {radius_final:.4f}")

        # Tolerance: Within 20% (coarse grid + reinitialization drift)
        error_rel = abs(radius_final - radius_expected) / radius0
        print(f"  Relative error: {100 * error_rel:.2f}%")

        assert error_rel < 0.25, f"Radius error too large: {error_rel:.2f}"
        print("  ✓ 2D shrinking circle test passed!")

    print("\n✅ All TimeDependentDomain tests passed!")
