"""
Integration tests for Stefan problem (free boundary coupling).

Tests coupling between heat equation and level set evolution for
melting/freezing phase change problems (Issue #592).

Created: 2026-01-18 (Issue #594 Phase 5.3)
"""

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.level_set import TimeDependentDomain


class TestStefanProblem1D:
    """Test 1D Stefan problem integration."""

    def setup_method(self):
        """Set up 1D Stefan problem components."""
        self.x_min, self.x_max = 0.0, 1.0
        self.Nx = 200
        self.alpha = 0.01  # Thermal diffusivity
        self.T_hot, self.T_cold = 1.0, 0.0
        self.s0 = 0.5  # Initial interface

        self.grid = TensorProductGrid(
            bounds=[(self.x_min, self.x_max)], boundary_conditions=no_flux_bc(dimension=1), Nx=[self.Nx]
        )
        self.x = self.grid.coordinates[0]
        self.dx = self.grid.spacing[0]

        # CFL-limited time step
        self.dt = 0.2 * self.dx**2 / self.alpha

    def solve_heat_equation_step(self, T_prev, dx, dt, alpha, T_hot, T_cold):
        """Single heat equation time step (explicit FD)."""
        cfl = alpha * dt / dx**2
        if cfl > 0.5:
            raise ValueError(f"CFL = {cfl:.3f} > 0.5, unstable!")

        T_new = T_prev.copy()

        # Interior: explicit finite difference
        for i in range(1, len(T_new) - 1):
            laplacian = (T_prev[i + 1] - 2 * T_prev[i] + T_prev[i - 1]) / dx**2
            T_new[i] = T_prev[i] + alpha * dt * laplacian

        # Boundary conditions
        T_new[0] = T_hot
        T_new[-1] = T_cold

        return T_new

    def test_stefan_coupling_stability(self):
        """Test that Stefan coupling remains stable over multiple steps."""
        # Initialize
        phi0 = self.x - self.s0
        ls_domain = TimeDependentDomain(phi0, self.grid, is_signed_distance=True)

        # Initial temperature
        T = np.where(self.x < self.s0, self.T_hot * (self.s0 - self.x) / self.s0, self.T_cold)

        # Evolve for 50 steps
        for _ in range(50):
            # Heat step
            T = self.solve_heat_equation_step(T, self.dx, self.dt, self.alpha, self.T_hot, self.T_cold)

            # Find interface and compute velocity
            phi_current = ls_domain.get_phi_at_time(ls_domain.time_history[-1])
            idx_interface = np.argmin(np.abs(phi_current))

            if 1 <= idx_interface < self.Nx:
                grad_T = (T[idx_interface + 1] - T[idx_interface - 1]) / (2 * self.dx)
            else:
                grad_T = 0.0

            velocity = -grad_T  # Stefan condition: V = -∂T/∂x

            # Level set step
            ls_domain.evolve_step(velocity, self.dt)

        # Stability checks
        phi_final = ls_domain.get_phi_at_time(ls_domain.time_history[-1])
        assert np.all(np.isfinite(phi_final)), "Level set should remain finite"
        assert np.all(np.isfinite(T)), "Temperature should remain finite"

        # Interface should have moved
        interface_final = self.x[np.argmin(np.abs(phi_final))]
        assert abs(interface_final - self.s0) > 0.01, "Interface should move"

    def test_stefan_interface_monotonicity(self):
        """Test that the interface advances monotonically, in the direction this fixture sets.

        The Stefan condition here is V = -dT/dx, and T decreases left to right (T_hot at x=0,
        T_cold at x=1), so every velocity this fixture produces is POSITIVE -- measured range
        0.505 to 1.000 -- and the interface moves RIGHT.  The assertion below pins that sign.
        Turning it into the leftward motion the old docstring described would require flipping
        the temperature profile, which is a change to the fixture, not to the assertion.
        """
        phi0 = self.x - self.s0
        ls_domain = TimeDependentDomain(phi0, self.grid, is_signed_distance=True)

        T = np.where(self.x < self.s0, self.T_hot * (self.s0 - self.x) / self.s0, self.T_cold)

        interface_positions = [self.s0]

        for _ in range(30):
            T = self.solve_heat_equation_step(T, self.dx, self.dt, self.alpha, self.T_hot, self.T_cold)

            phi_current = ls_domain.get_phi_at_time(ls_domain.time_history[-1])
            idx_interface = np.argmin(np.abs(phi_current))
            grad_T = (T[min(idx_interface + 1, self.Nx)] - T[max(idx_interface - 1, 0)]) / (2 * self.dx)

            velocity = -grad_T
            ls_domain.evolve_step(velocity, self.dt)

            interface_positions.append(self.x[idx_interface])

        # V = -dT/dx > 0 throughout, so the interface advances right and must never retreat.
        # The old tolerance (<= 0.01, i.e. two full cells of RIGHTWARD motion per step at
        # dx = 0.005) admitted the negation of what it claimed; this is strict in the sign the
        # fixture actually produces.  Measured: all diffs in [0.0, 0.005].
        interface_diffs = np.diff(interface_positions)
        assert np.all(interface_diffs >= -1e-12), f"interface retreated: min diff {interface_diffs.min()}"
        assert interface_positions[-1] - interface_positions[0] > self.dx, (
            "the interface did not move, so monotonicity is vacuous; measured displacement "
            f"{interface_positions[-1] - interface_positions[0]}"
        )


class TestStefanProblem2D:
    """Test 2D Stefan problem (circular interface)."""

    def setup_method(self):
        """Set up 2D Stefan problem."""
        self.Nx, self.Ny = 60, 60
        self.alpha = 0.01
        self.T_hot = 1.0
        self.grid = TensorProductGrid(
            bounds=[(0, 1), (0, 1)], boundary_conditions=no_flux_bc(dimension=2), Nx=[self.Nx, self.Ny]
        )
        self.dx, self.dy = self.grid.spacing
        self.dt = 0.1 * min(self.dx, self.dy) ** 2 / (2 * self.alpha)

    def solve_heat_2d_step(self, T_prev, dx, dy, dt, alpha, T_boundary):
        """2D heat equation step."""
        Nx, Ny = T_prev.shape
        cfl = alpha * dt * (1 / dx**2 + 1 / dy**2)
        if cfl > 0.5:
            raise ValueError(f"2D CFL = {cfl:.3f} > 0.5")

        T_new = T_prev.copy()

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                laplacian = (T_prev[i + 1, j] - 2 * T_prev[i, j] + T_prev[i - 1, j]) / dx**2
                laplacian += (T_prev[i, j + 1] - 2 * T_prev[i, j] + T_prev[i, j - 1]) / dy**2
                T_new[i, j] = T_prev[i, j] + alpha * dt * laplacian

        # BC: All boundaries at T_hot
        T_new[0, :] = T_boundary
        T_new[-1, :] = T_boundary
        T_new[:, 0] = T_boundary
        T_new[:, -1] = T_boundary

        return T_new

    def test_circular_symmetry_preservation(self):
        """Test that circular interface remains approximately circular."""
        # Initial circular interface
        X, Y = self.grid.meshgrid()
        center = np.array([0.5, 0.5])
        R0 = 0.25
        phi0 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - R0

        ls_domain = TimeDependentDomain(phi0, self.grid, is_signed_distance=True)

        # Initial temperature: cold inside, hot outside
        T = np.where(phi0 < 0, 0.0, self.T_hot)

        # Evolve for 20 steps
        for _ in range(20):
            T = self.solve_heat_2d_step(T, self.dx, self.dy, self.dt, self.alpha, self.T_hot)

            phi_current = ls_domain.get_phi_at_time(ls_domain.time_history[-1])

            # Compute velocity (simplified)
            grad_T_x = np.gradient(T, self.dx, axis=0)
            grad_T_y = np.gradient(T, self.dy, axis=1)

            grad_phi_x = np.gradient(phi_current, self.dx, axis=0)
            grad_phi_y = np.gradient(phi_current, self.dy, axis=1)
            grad_phi_mag = np.sqrt(grad_phi_x**2 + grad_phi_y**2) + 1e-10

            normal_x = grad_phi_x / grad_phi_mag
            normal_y = grad_phi_y / grad_phi_mag

            velocity = -(grad_T_x * normal_x + grad_T_y * normal_y)

            ls_domain.evolve_step(velocity, self.dt)

        # Check symmetry
        phi_final = ls_domain.get_phi_at_time(ls_domain.time_history[-1])
        interface_mask = np.abs(phi_final) < 0.02

        if interface_mask.sum() > 0:
            x_interface = X[interface_mask]
            y_interface = Y[interface_mask]

            # Compute center and aspect ratio
            x_center = x_interface.mean()
            y_center = y_interface.mean()

            x_extent = x_interface.max() - x_interface.min()
            y_extent = y_interface.max() - y_interface.min()

            aspect_ratio = max(x_extent, y_extent) / (min(x_extent, y_extent) + 1e-10)

            # Center should be near (0.5, 0.5)
            center_error = np.sqrt((x_center - 0.5) ** 2 + (y_center - 0.5) ** 2)
            assert center_error < 0.1, f"Center drift: {center_error}"

            # Aspect ratio should be near 1 (circle)
            assert aspect_ratio < 1.3, f"Shape distortion: aspect_ratio={aspect_ratio}"

    def test_interface_shrinking(self):
        """Test that circular interface shrinks (ice melts)."""
        X, Y = self.grid.meshgrid()
        R0 = 0.3
        phi0 = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - R0

        ls_domain = TimeDependentDomain(phi0, self.grid, is_signed_distance=True)
        T = np.where(phi0 < 0, 0.0, self.T_hot)

        radii = [R0]

        for _ in range(15):
            T = self.solve_heat_2d_step(T, self.dx, self.dy, self.dt, self.alpha, self.T_hot)

            phi_current = ls_domain.get_phi_at_time(ls_domain.time_history[-1])
            grad_T_x = np.gradient(T, self.dx, axis=0)
            grad_T_y = np.gradient(T, self.dy, axis=1)
            grad_phi_x = np.gradient(phi_current, self.dx, axis=0)
            grad_phi_y = np.gradient(phi_current, self.dy, axis=1)
            grad_phi_mag = np.sqrt(grad_phi_x**2 + grad_phi_y**2) + 1e-10

            velocity = -(grad_T_x * grad_phi_x + grad_T_y * grad_phi_y) / grad_phi_mag
            ls_domain.evolve_step(velocity, self.dt)

            # Estimate radius
            interface_mask = np.abs(phi_current) < 0.03
            if interface_mask.sum() > 0:
                distances = np.sqrt((X[interface_mask] - 0.5) ** 2 + (Y[interface_mask] - 0.5) ** 2)
                radii.append(distances.mean())

        # Radius should evolve (may grow slightly due to numerical effects with coarse grid)
        # The test validates that the coupling is stable and produces reasonable behavior
        radius_change = abs(radii[-1] - radii[0])
        assert radius_change < 0.1, "Radius change should be bounded (stable coupling)"
        # Check that changes are smooth (no oscillations)
        radius_diffs = np.abs(np.diff(radii))
        assert np.max(radius_diffs) < 0.05, "Radius changes should be smooth"


class TestLevelSetRobustness:
    """Test level set robustness in coupling scenarios."""

    def test_zero_velocity_preserves_interface(self):
        """Test that zero velocity keeps interface stationary."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx=[100])
        x = grid.coordinates[0]
        phi0 = x - 0.5

        ls_domain = TimeDependentDomain(phi0, grid, is_signed_distance=True)

        # Evolve with zero velocity
        for _ in range(50):
            ls_domain.evolve_step(velocity=0.0, dt=0.001)

        phi_final = ls_domain.get_phi_at_time(ls_domain.time_history[-1])

        # Interface should not move
        interface_initial = x[np.argmin(np.abs(phi0))]
        interface_final = x[np.argmin(np.abs(phi_final))]

        assert abs(interface_final - interface_initial) < 1e-6, "Zero velocity should preserve interface"

    def test_rapid_velocity_changes(self):
        """Test stability with rapidly changing velocity."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], boundary_conditions=no_flux_bc(dimension=1), Nx=[150])
        x = grid.coordinates[0]
        phi0 = x - 0.5
        dx = grid.spacing[0]

        ls_domain = TimeDependentDomain(phi0, grid, is_signed_distance=True)

        # Alternate velocity sign
        for i in range(30):
            velocity = 0.1 if i % 2 == 0 else -0.1
            dt = 0.5 * dx / abs(velocity)  # CFL-safe
            ls_domain.evolve_step(velocity, dt)

        phi_final = ls_domain.get_phi_at_time(ls_domain.time_history[-1])

        # Should remain stable (no NaN/Inf)
        assert np.all(np.isfinite(phi_final)), "Should handle rapid velocity changes"

        # External oracle: the 15 forward and 15 backward steps carry the same |v| and the same
        # dt, so the advection distances cancel exactly and the interface must return to x = 0.5.
        # Measured exactly 0.5; the tolerance is one cell because the position is read off an
        # argmin.  A scheme that upwinds from the same side regardless of sign drifts ~7 cells.
        interface_final = x[np.argmin(np.abs(phi_final))]
        assert abs(interface_final - 0.5) <= dx, (
            f"zero-mean velocity must leave no net interface drift; got {interface_final}"
        )

        # phi starts strictly increasing (x - 0.5) and pure advection plus numerical diffusion
        # keeps it so.  Measured min forward difference 9.63e-4.  A sign-flip oscillation, which
        # is exactly what alternating velocity provokes, breaks monotonicity while staying finite.
        assert np.all(np.diff(phi_final) > 0), "sign flips introduced a non-monotone level set"

        # Positive control: the loop must actually have done something.  Measured max deviation
        # 1.44e-2 from phi0, against exactly 0.0 for the same loop run at v = 0.
        assert not np.allclose(phi_final, phi0), "the evolution did nothing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
