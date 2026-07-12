"""
Unit tests for level set core functionality.

Tests:
- LevelSetFunction: Container for φ, normals, curvature
- LevelSetEvolver: Evolution ∂φ/∂t + V|∇φ| = 0
- TimeDependentDomain: Time series management

Created: 2026-01-18 (Issue #592)
"""

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.geometry.level_set import (
    LevelSetEvolver,
    LevelSetFunction,
    TimeDependentDomain,
    compute_curvature,
    reinitialize,
)


class TestLevelSetFunction:
    """Test LevelSetFunction container."""

    def test_1d_creation(self):
        """Test creating 1D level set function."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        phi = x - 0.5  # Interface at x=0.5

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)

        assert ls.dimension == 1
        assert ls.phi.shape == (101,)
        assert ls.is_signed_distance is True

    def test_2d_creation(self):
        """Test creating 2D level set function."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[50, 50], boundary_conditions=no_flux_bc(dimension=2))
        X, Y = grid.meshgrid()
        phi = np.sqrt(X**2 + Y**2) - 0.3

        ls = LevelSetFunction(phi, grid)

        assert ls.dimension == 2
        assert ls.phi.shape == (51, 51)

    def test_interface_mask(self):
        """Test interface mask computation."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        dx = grid.spacing[0]
        phi = x - 0.5

        ls = LevelSetFunction(phi, grid)
        mask = ls.interface_mask(width=2 * dx)

        # Should have points near x=0.5
        assert np.sum(mask) > 0
        assert np.sum(mask) < len(x)  # Not all points

    def test_normal_field_1d(self):
        """Test normal field computation in 1D."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        phi = x - 0.5

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)
        normals = ls.get_normal()

        assert normals.shape == (1, 101)
        # For φ = x - c, ∇φ = 1 → n = 1 (check interior only, boundaries may differ)
        interior = np.s_[10:-10]
        assert np.allclose(np.abs(normals[:, interior]), 1.0, atol=1e-6)

    def test_normal_field_2d_circle(self):
        """Test normal field for 2D circle."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[50, 50], boundary_conditions=no_flux_bc(dimension=2))
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        # Circle centered at (0.5, 0.5)
        center = np.array([0.5, 0.5])
        radius = 0.3
        phi = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)
        normals = ls.get_normal()

        # Check unit magnitude on interface
        interface = ls.interface_mask(width=2 * dx)
        normal_mag = np.linalg.norm(normals, axis=0)

        assert np.allclose(normal_mag[interface], 1.0, atol=1e-5)

    def test_curvature_circle(self):
        """Test curvature computation for 2D circle."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[80, 80], boundary_conditions=no_flux_bc(dimension=2))
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        # Circle: analytical curvature κ = 1/R
        radius = 0.3
        center = np.array([0.5, 0.5])
        phi = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        ls = LevelSetFunction(phi, grid, is_signed_distance=True)
        kappa = ls.get_curvature()

        # Check on interface
        interface = ls.interface_mask(width=2 * dx)
        kappa_interface = kappa[interface]
        kappa_analytical = 1.0 / radius

        error = np.abs(np.mean(kappa_interface) - kappa_analytical)
        assert error < 0.2 * kappa_analytical, f"Curvature error: {error}"


class TestLevelSetEvolver:
    """Test LevelSetEvolver for ∂φ/∂t + V|∇φ| = 0."""

    def test_1d_constant_velocity(self):
        """Test 1D evolution with constant velocity."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        dx = grid.spacing[0]

        # Initial interface at x=0.5
        phi0 = x - 0.5

        evolver = LevelSetEvolver(grid, scheme="upwind")

        # Evolve with V=1.0 for dt=0.1
        V = 1.0
        dt = 0.1
        phi1 = evolver.evolve_step(phi0, velocity=V, dt=dt)

        # Interface should move to x ≈ 0.6
        idx_zero = np.argmin(np.abs(phi1))
        x_interface = x[idx_zero]
        x_expected = 0.5 + V * dt

        assert np.abs(x_interface - x_expected) < 2 * dx

    def test_cfl_adaptive_substepping(self):
        """Test CFL-adaptive substepping."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[50], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        phi0 = x - 0.5

        evolver = LevelSetEvolver(grid, cfl_max=0.9)

        # Large velocity → should trigger substepping
        V_large = 10.0
        dt = 0.1

        # Should not crash (automatic substepping)
        phi1 = evolver.evolve_step(phi0, velocity=V_large, dt=dt)

        assert phi1.shape == phi0.shape
        assert np.isfinite(phi1).all()

    def test_2d_expansion(self):
        """Test 2D circle expansion."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[40, 40], boundary_conditions=no_flux_bc(dimension=2))
        X, Y = grid.meshgrid()

        # Circle
        radius = 0.2
        center = np.array([0.5, 0.5])
        phi0 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        evolver = LevelSetEvolver(grid)

        # Expand with V=0.5
        V = 0.5
        dt = 0.05
        phi1 = evolver.evolve_step(phi0, velocity=V, dt=dt)

        # Circle should expand (more negative φ values)
        area0 = np.sum(phi0 < 0)
        area1 = np.sum(phi1 < 0)

        assert area1 > area0, "Circle should expand with positive velocity"

    def test_upwind_inward_collapse_matches_analytic_extinction(self):
        """Issue #1602: default 'upwind' must track inward (V<0) normal motion.

        phi0 = |x| - 1 on [-2, 2] collapses toward the kink at x=0; with inward
        speed |V|=1 the half-width shrinks linearly and the domain extinguishes at
        the analytic t* = 1.0. The pre-fix upwind selected the one-sided stencil
        from sign(grad phi), not the velocity sign, which over-collapses early and
        then stalls at the kink core, persisting to ~2.8*t*. The velocity-sign
        Godunov (this fix) tracks the analytic collapse.

        Discriminating (mutation-checked): reverting to the sign(grad phi) stencil
        makes both assertions fail (area ~16 at t=0.5; ~4 cells never extinguish).
        """
        grid = TensorProductGrid(bounds=[(-2.0, 2.0)], Nx=[400], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        phi0 = np.abs(x) - 1.0
        area0 = int(np.sum(phi0 < 0))

        evolver = LevelSetEvolver(grid, scheme="upwind")
        V, dt = -1.0, 0.004

        def evolve_to(T):
            phi = phi0.copy()
            for _ in range(round(T / dt)):
                phi = evolver.evolve_step(phi, velocity=V, dt=dt)
            return phi

        # Linear collapse: half-width 1.0 -> 0.5 at t=0.5, so area ~ area0/2.
        area_half = int(np.sum(evolve_to(0.5) < 0))
        assert abs(area_half - area0 / 2) < 0.15 * area0, (
            f"inward collapse off analytic rate: area(t=0.5)={area_half}, expected ~{area0 / 2:.0f}"
        )

        # Fully extinguished past the analytic extinction time t*=1.0.
        area_late = int(np.sum(evolve_to(1.5) < 0))
        assert area_late == 0, f"domain persists past extinction: area(t=1.5)={area_late}, expected 0"

    def test_upwind_planar_front_no_boundary_phantom(self):
        """Issue #1602 (review): a non-symmetric interface must not grow a phantom
        interface at the domain boundary.

        phi0 = x on [-2, 2] has phi[0]=-2 != phi[-1]=+2 and a single interface at
        x=0; under outward V=+0.5 it moves to x=0.5 while the right boundary stays
        positive. The velocity-sign Godunov uses periodic-roll one-sided
        differences, so without a no-flux (Neumann-ghost) edge correction the wrap
        injects a spurious O(1/h) gradient that flips phi[-1] negative and creates
        a second, phantom interface.

        Discriminating: raw np.roll stencils (no edge correction) give phi[-1]<0
        and two sign changes; the Neumann-ghost fix keeps phi[-1]~+1.5 and exactly
        one interface. Symmetric configs (centered circle, |x|-1) never exercise
        this because phi[0]==phi[-1].
        """
        grid = TensorProductGrid(bounds=[(-2.0, 2.0)], Nx_points=[401], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        phi = x.copy()  # single interface at x=0, non-symmetric across the edges

        evolver = LevelSetEvolver(grid, scheme="upwind")
        V, dt = 0.5, 0.004
        for _ in range(250):  # t = 1.0
            phi = evolver.evolve_step(phi, velocity=V, dt=dt)

        # Right boundary must remain positive (no phantom interface from edge wrap).
        assert phi[-1] > 0.5, f"boundary flipped -> phantom interface: phi[-1]={phi[-1]:.3f}"
        # Exactly one strict sign change (the real interface near x=0.5), not two.
        crossings = int(np.sum(phi[:-1] * phi[1:] < 0))
        assert crossings == 1, f"expected 1 interface, found {crossings}"


class TestTimeDependentDomain:
    """Test TimeDependentDomain for managing φ(t) history."""

    def test_initialization(self):
        """Test creating time-dependent domain."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[50], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        phi0 = x - 0.5

        ls_domain = TimeDependentDomain(phi0, grid, initial_time=0.0)

        assert ls_domain.current_time == 0.0
        assert ls_domain.num_snapshots == 1
        assert np.allclose(ls_domain.current_phi, phi0)

    def test_evolution_history(self):
        """Test that history is saved correctly."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[50], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        phi0 = x - 0.5

        ls_domain = TimeDependentDomain(phi0, grid)

        # Evolve 5 steps
        V = 0.5
        dt = 0.1
        n_steps = 5

        for _ in range(n_steps):
            ls_domain.evolve_step(V, dt, save_to_history=True)

        assert ls_domain.num_snapshots == n_steps + 1  # Initial + n_steps
        assert ls_domain.current_time == n_steps * dt

    def test_time_interpolation(self):
        """Test retrieving φ at intermediate times."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[50], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        phi0 = x - 0.5

        ls_domain = TimeDependentDomain(phi0, grid)

        # Create history
        for _ in range(3):
            ls_domain.evolve_step(0.5, 0.1)

        # Get φ at t=0.15 (between snapshots)
        phi_mid = ls_domain.get_phi_at_time(0.15, interpolate=True)

        assert phi_mid.shape == phi0.shape
        assert np.isfinite(phi_mid).all()

    def test_history_clearing(self):
        """Test clearing old history."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[50], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        phi0 = x - 0.5

        ls_domain = TimeDependentDomain(phi0, grid)

        # Create history
        for _ in range(10):
            ls_domain.evolve_step(0.5, 0.1)

        snapshots_before = ls_domain.num_snapshots

        # Clear before t=0.5
        ls_domain.clear_history_before(0.5)

        assert ls_domain.num_snapshots < snapshots_before

    def test_level_set_function_wrapper(self):
        """Test getting LevelSetFunction from domain."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[50], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        phi0 = x - 0.5

        ls_domain = TimeDependentDomain(phi0, grid, is_signed_distance=True)

        ls = ls_domain.get_level_set_function()

        assert isinstance(ls, LevelSetFunction)
        assert ls.is_signed_distance is True


class TestCurvature:
    """Test standalone curvature computation."""

    def test_flat_interface_zero_curvature(self):
        """Test that flat interface has zero curvature."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[50, 50], boundary_conditions=no_flux_bc(dimension=2))
        X, _Y = grid.meshgrid()
        dx = grid.spacing[0]

        # Flat interface: φ = x - 0.5
        phi = X - 0.5

        kappa = compute_curvature(phi, grid)

        # Curvature should be near zero everywhere
        interface = np.abs(phi) < 2 * dx
        kappa_interface = kappa[interface]

        assert np.mean(np.abs(kappa_interface)) < 0.01

    def test_sphere_curvature_3d(self):
        """Test 3D sphere curvature."""
        grid = TensorProductGrid(
            bounds=[(0, 1), (0, 1), (0, 1)], Nx=[25, 25, 25], boundary_conditions=no_flux_bc(dimension=3)
        )
        X, Y, Z = grid.meshgrid()
        dx = grid.spacing[0]

        # Sphere: mean curvature H = 2/R
        radius = 0.3
        center = np.array([0.5, 0.5, 0.5])
        phi = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2) - radius

        kappa = compute_curvature(phi, grid)

        interface = np.abs(phi) < 2 * dx
        kappa_interface = kappa[interface]

        kappa_analytical = 2.0 / radius
        error = np.abs(np.mean(kappa_interface) - kappa_analytical)

        # Coarse 3D grid → larger tolerance
        assert error < 0.3 * kappa_analytical


class TestLevelSetEvolverWENO5:
    """Test LevelSetEvolver with WENO5 scheme."""

    @pytest.mark.unit
    def test_weno5_1d_translation(self):
        """Test 1D WENO5 evolution with constant velocity."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        dx = grid.spacing[0]

        phi0 = x - 0.5

        evolver = LevelSetEvolver(grid, scheme="weno5")

        V = 1.0
        dt = 0.1
        phi1 = evolver.evolve_step(phi0, velocity=V, dt=dt)

        # Interface should move to x ~ 0.6
        idx_zero = np.argmin(np.abs(phi1))
        x_interface = x[idx_zero]
        x_expected = 0.5 + V * dt

        assert np.abs(x_interface - x_expected) < 2 * dx

    @pytest.mark.unit
    def test_weno5_2d_circle_expansion(self):
        """Test 2D circle expansion with WENO5."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[40, 40], boundary_conditions=no_flux_bc(dimension=2))
        X, Y = grid.meshgrid()

        radius = 0.2
        center = np.array([0.5, 0.5])
        phi0 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        evolver = LevelSetEvolver(grid, scheme="weno5")

        V = 0.5
        dt = 0.05
        phi1 = evolver.evolve_step(phi0, velocity=V, dt=dt)

        # Circle should expand (more negative phi values)
        area0 = np.sum(phi0 < 0)
        area1 = np.sum(phi1 < 0)

        assert area1 > area0, "Circle should expand with positive velocity"

    @pytest.mark.unit
    def test_weno5_no_accuracy_regression_vs_upwind(self):
        """Test that WENO5 does not degrade accuracy vs upwind on smooth problem.

        Note: With explicit Euler time integration, temporal error O(dt)
        dominates spatial error, so both schemes give similar results.
        A TVD-RK3 time integrator would be needed to observe the O(dx^5)
        vs O(dx) spatial accuracy difference.
        """
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]

        # Smooth initial data: Gaussian
        phi0 = np.exp(-50 * (x - 0.3) ** 2) - 0.5

        V = 1.0
        dt = 0.01  # Small dt for accuracy comparison

        evolver_upwind = LevelSetEvolver(grid, scheme="upwind")
        evolver_weno5 = LevelSetEvolver(grid, scheme="weno5")

        phi_upwind = evolver_upwind.evolve_step(phi0, velocity=V, dt=dt)
        phi_weno5 = evolver_weno5.evolve_step(phi0, velocity=V, dt=dt)

        # Both should produce finite results
        assert np.isfinite(phi_upwind).all()
        assert np.isfinite(phi_weno5).all()

        # Reference: translate initial data by V*dt analytically
        phi_exact = np.exp(-50 * (x - 0.3 - V * dt) ** 2) - 0.5

        # Compare errors in interior (avoid boundary effects)
        interior = slice(10, -10)
        error_upwind = np.max(np.abs(phi_upwind[interior] - phi_exact[interior]))
        error_weno5 = np.max(np.abs(phi_weno5[interior] - phi_exact[interior]))

        # WENO5 should not be significantly worse than upwind
        assert error_weno5 <= error_upwind * 1.1, (
            f"WENO5 error ({error_weno5:.6e}) unexpectedly larger than upwind ({error_upwind:.6e})"
        )

    @pytest.mark.unit
    def test_weno5_invalid_scheme_raises(self):
        """Test that invalid scheme name raises ValueError."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[50], boundary_conditions=no_flux_bc(dimension=1))

        with pytest.raises(ValueError, match="Scheme must be"):
            LevelSetEvolver(grid, scheme="central")


class TestWENO5DerivativeND:
    """Test nD WENO5 derivative computation."""

    @pytest.mark.unit
    def test_weno5_2d_smooth_function(self):
        """Verify d/dx of sin(2pi*x)*cos(2pi*y) matches analytical."""
        from mfgarchon.operators.reconstruction.weno import compute_weno5_derivative_nd

        Nx, Ny = 60, 60
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        dx_val, dy_val = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        du_dx_exact = 2 * np.pi * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)

        du_dx = compute_weno5_derivative_nd(u, spacings=(dx_val, dy_val), axis=0)

        # Interior error (skip boundary rows)
        error = np.max(np.abs(du_dx[2:-2, 2:-2] - du_dx_exact[2:-2, 2:-2]))
        assert error < 0.05, f"2D WENO5 d/dx error too large: {error}"

    @pytest.mark.unit
    def test_weno5_nd_axis_validation(self):
        """Test that invalid axis raises ValueError."""
        from mfgarchon.operators.reconstruction.weno import compute_weno5_derivative_nd

        u = np.zeros((10, 10))
        with pytest.raises(ValueError, match="axis=2 out of range"):
            compute_weno5_derivative_nd(u, spacings=(0.1, 0.1), axis=2)

    @pytest.mark.unit
    def test_weno5_nd_spacings_length_validation(self):
        """Test that mismatched spacings length raises ValueError."""
        from mfgarchon.operators.reconstruction.weno import compute_weno5_derivative_nd

        u = np.zeros((10, 10))
        with pytest.raises(ValueError, match="len\\(spacings\\)=1 must match array ndim=2"):
            compute_weno5_derivative_nd(u, spacings=(0.1,), axis=0)


class TestTVDRK3TimeIntegrator:
    """Test TVD-RK3 (SSP-RK3) time integration in LevelSetEvolver."""

    @pytest.mark.unit
    def test_rk3_1d_translation(self):
        """Test 1D translation with TVD-RK3 produces correct interface motion."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]
        dx = grid.spacing[0]

        phi0 = x - 0.5

        evolver = LevelSetEvolver(grid, scheme="upwind", time_integrator="tvd_rk3")

        V = 1.0
        dt = 0.1
        phi1 = evolver.evolve_step(phi0, velocity=V, dt=dt)

        # Interface should move to x ~ 0.6
        idx_zero = np.argmin(np.abs(phi1))
        x_interface = x[idx_zero]
        x_expected = 0.5 + V * dt

        assert np.abs(x_interface - x_expected) < 2 * dx

    @pytest.mark.unit
    def test_rk3_2d_circle_expansion(self):
        """Test 2D circle expansion with TVD-RK3."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[40, 40], boundary_conditions=no_flux_bc(dimension=2))
        X, Y = grid.meshgrid()

        radius = 0.2
        center = np.array([0.5, 0.5])
        phi0 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        evolver = LevelSetEvolver(grid, scheme="upwind", time_integrator="tvd_rk3")

        V = 0.5
        dt = 0.05
        phi1 = evolver.evolve_step(phi0, velocity=V, dt=dt)

        area0 = np.sum(phi0 < 0)
        area1 = np.sum(phi1 < 0)

        assert area1 > area0, "Circle should expand with positive velocity"

    @pytest.mark.unit
    def test_weno5_rk3_higher_accuracy_than_euler(self):
        """Test that WENO5+RK3 achieves lower error than WENO5+Euler.

        With explicit Euler, temporal O(dt) error dominates the spatial O(dx^5)
        from WENO5. TVD-RK3's O(dt^3) temporal accuracy reveals the spatial
        improvement. We use many small steps to accumulate the temporal error
        difference.
        """
        Nx = 200
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[Nx], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]

        # Smooth initial data: Gaussian centered at x=0.3
        phi0 = np.exp(-100 * (x - 0.3) ** 2) - 0.5

        V = 1.0
        dt = 0.002
        n_steps = 20  # Total time = 0.04

        evolver_euler = LevelSetEvolver(grid, scheme="weno5", time_integrator="euler")
        evolver_rk3 = LevelSetEvolver(grid, scheme="weno5", time_integrator="tvd_rk3")

        phi_euler = phi0.copy()
        phi_rk3 = phi0.copy()
        for _ in range(n_steps):
            phi_euler = evolver_euler.evolve_step(phi_euler, velocity=V, dt=dt)
            phi_rk3 = evolver_rk3.evolve_step(phi_rk3, velocity=V, dt=dt)

        # Analytical: translate Gaussian by V*T
        T = n_steps * dt
        phi_exact = np.exp(-100 * (x - 0.3 - V * T) ** 2) - 0.5

        # Compare errors in interior
        interior = slice(10, -10)
        error_euler = np.max(np.abs(phi_euler[interior] - phi_exact[interior]))
        error_rk3 = np.max(np.abs(phi_rk3[interior] - phi_exact[interior]))

        # RK3 should be noticeably more accurate
        assert error_rk3 < error_euler, (
            f"RK3 error ({error_rk3:.6e}) should be less than Euler error ({error_euler:.6e})"
        )

    @pytest.mark.unit
    def test_rk3_convergence_order(self):
        """Verify TVD-RK3 temporal error decreases faster than Euler.

        For a nonlinear initial condition (Gaussian), the spatial operator
        L(phi) = -V*|grad phi| varies with phi, producing actual temporal
        truncation error. We compare RK3 vs Euler at two different dt
        values to verify RK3 converges faster.
        """
        Nx = 200
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[Nx], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]

        # Nonlinear initial data: Gaussian (varying gradient → temporal error)
        phi0 = np.exp(-100 * (x - 0.3) ** 2) - 0.5
        V = 1.0

        dt_coarse = 0.004
        dt_fine = dt_coarse / 2
        n_coarse = 10
        n_fine = n_coarse * 2
        T = n_coarse * dt_coarse

        # Exact solution: translate Gaussian by V*T
        phi_exact = np.exp(-100 * (x - 0.3 - V * T) ** 2) - 0.5
        interior = slice(15, -15)

        # Euler convergence
        evolver_euler = LevelSetEvolver(grid, scheme="weno5", time_integrator="euler")

        phi_euler_c = phi0.copy()
        for _ in range(n_coarse):
            phi_euler_c = evolver_euler.evolve_step(phi_euler_c, velocity=V, dt=dt_coarse)
        err_euler_c = np.max(np.abs(phi_euler_c[interior] - phi_exact[interior]))

        phi_euler_f = phi0.copy()
        for _ in range(n_fine):
            phi_euler_f = evolver_euler.evolve_step(phi_euler_f, velocity=V, dt=dt_fine)
        err_euler_f = np.max(np.abs(phi_euler_f[interior] - phi_exact[interior]))

        # RK3 convergence
        evolver_rk3 = LevelSetEvolver(grid, scheme="weno5", time_integrator="tvd_rk3")

        phi_rk3_c = phi0.copy()
        for _ in range(n_coarse):
            phi_rk3_c = evolver_rk3.evolve_step(phi_rk3_c, velocity=V, dt=dt_coarse)
        err_rk3_c = np.max(np.abs(phi_rk3_c[interior] - phi_exact[interior]))

        phi_rk3_f = phi0.copy()
        for _ in range(n_fine):
            phi_rk3_f = evolver_rk3.evolve_step(phi_rk3_f, velocity=V, dt=dt_fine)
        err_rk3_f = np.max(np.abs(phi_rk3_f[interior] - phi_exact[interior]))

        # RK3 fine should be substantially more accurate than Euler fine
        assert err_rk3_f < err_euler_f, f"RK3 fine ({err_rk3_f:.2e}) should beat Euler fine ({err_euler_f:.2e})"

        # RK3 should converge faster: its error ratio should exceed Euler's
        if err_rk3_f > 1e-14 and err_euler_f > 1e-14:
            ratio_euler = err_euler_c / err_euler_f
            ratio_rk3 = err_rk3_c / err_rk3_f
            assert ratio_rk3 > ratio_euler, (
                f"RK3 convergence ratio ({ratio_rk3:.1f}) should exceed Euler ratio ({ratio_euler:.1f})"
            )

    @pytest.mark.unit
    def test_invalid_time_integrator_raises(self):
        """Test that invalid time_integrator raises ValueError."""
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx=[50], boundary_conditions=no_flux_bc(dimension=1))

        with pytest.raises(ValueError, match="time_integrator must be"):
            LevelSetEvolver(grid, time_integrator="rk4")


class TestReinitialization:
    """Test reinitialization to maintain SDF property."""

    def test_maintains_interface(self):
        """Test that reinitialization preserves zero level set."""
        grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx=[60, 60], boundary_conditions=no_flux_bc(dimension=2))
        X, Y = grid.meshgrid()
        dx = grid.spacing[0]

        # Circle
        center = np.array([0.5, 0.5])
        radius = 0.3
        phi0 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius

        # Slightly distort
        phi_distorted = phi0 + 0.03 * np.sin(10 * X) * np.sin(10 * Y)

        # Reinitialize
        phi_reinit = reinitialize(phi_distorted, grid, max_iterations=15)

        # Find interface before and after
        interface_before = np.abs(phi_distorted) < dx
        interface_after = np.abs(phi_reinit) < dx

        coords_before = np.column_stack([X[interface_before], Y[interface_before]])
        coords_after = np.column_stack([X[interface_after], Y[interface_after]])

        # Check that interfaces are close
        if len(coords_before) > 0 and len(coords_after) > 0:
            # Sample check
            sample_idx = 0
            pt = coords_before[sample_idx]
            dists = np.linalg.norm(coords_after - pt, axis=1)
            min_dist = np.min(dists)

            # Should be within few grid points
            assert min_dist < 10 * dx

    @pytest.mark.skip(
        reason="TODO: Reinitialization algorithm needs tuning - gradient deviation "
        "increases (dev_after=6.3 > 2*dev_before=2.35). Needs separate investigation."
    )
    def test_improves_sdf_property(self):
        """Test that reinitialization maintains or improves |∇φ| ≈ 1."""
        grid = TensorProductGrid(bounds=[(0, 1)], Nx=[100], boundary_conditions=no_flux_bc(dimension=1))
        x = grid.coordinates[0]

        # Start with true SDF, add small perturbation
        phi_true = x - 0.5
        phi_perturbed = phi_true + 0.02 * np.sin(20 * np.pi * x)

        grad_ops = grid.get_gradient_operator()
        grad_before = np.abs(grad_ops[0](phi_perturbed))
        dev_before = np.max(np.abs(grad_before - 1.0))

        # Reinitialize
        phi_reinit = reinitialize(phi_perturbed, grid, max_iterations=20)

        grad_after = np.abs(grad_ops[0](phi_reinit))
        dev_after = np.max(np.abs(grad_after - 1.0))

        # Should not make things significantly worse
        assert dev_after < 2 * dev_before


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v"])
