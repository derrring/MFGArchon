#!/usr/bin/env python3
"""
Unit tests for WENO Family HJB Solver

Tests the unified WENO family solver implementation with strategic typing excellence.
Validates:
1. All WENO variants are accessible and functional
2. Parameter validation works correctly
3. Weight computations are mathematically sound
4. Interface consistency across variants
5. Strategic typing compliance

Usage:
    python -m pytest tests/unit/test_weno_family.py -v
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBWenoSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


class TestWenoFamilySolver:
    """Test suite for WENO Family HJB Solver."""

    @pytest.fixture
    def simple_problem(self) -> MFGProblem:
        """Create simple MFG problem for testing using modern geometry-first API."""
        domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[33], boundary_conditions=no_flux_bc(dimension=1))
        return MFGProblem(geometry=domain, T=0.1, Nt=10, sigma=0.1, components=_default_components())

    @pytest.fixture
    def test_values(self) -> np.ndarray:
        """Create test array for WENO weight computation."""
        # Simple polynomial for testing: u(x) = x^2
        x = np.linspace(0, 1, 21)
        return x**2

    def test_all_variants_available(self, simple_problem):
        """Test that all WENO variants can be instantiated."""
        variants = ["weno5", "weno-z", "weno-m", "weno-js"]

        for variant in variants:
            solver = HJBWenoSolver(problem=simple_problem, weno_variant=variant)
            assert solver.weno_variant == variant
            assert variant.upper() in solver.hjb_method_name

    def test_invalid_variant_raises_error(self, simple_problem):
        """Test that invalid WENO variant raises ValueError."""
        with pytest.raises(ValueError, match="Unknown WENO variant"):
            HJBWenoSolver(problem=simple_problem, weno_variant="invalid_variant")

    def test_parameter_validation(self, simple_problem):
        """Test parameter validation for all solver parameters."""

        # Test invalid CFL number
        with pytest.raises(ValueError, match="CFL number must be in"):
            HJBWenoSolver(simple_problem, cfl_number=0.0)

        with pytest.raises(ValueError, match="CFL number must be in"):
            HJBWenoSolver(simple_problem, cfl_number=1.5)

        # Test invalid diffusion stability factor
        with pytest.raises(ValueError, match="Diffusion stability factor"):
            HJBWenoSolver(simple_problem, diffusion_stability_factor=0.0)

        with pytest.raises(ValueError, match="Diffusion stability factor"):
            HJBWenoSolver(simple_problem, diffusion_stability_factor=0.6)

        # Test invalid WENO epsilon
        with pytest.raises(ValueError, match="WENO epsilon must be positive"):
            HJBWenoSolver(simple_problem, weno_epsilon=0.0)

        # Test invalid WENO-Z parameter
        with pytest.raises(ValueError, match="WENO-Z parameter"):
            HJBWenoSolver(simple_problem, weno_z_parameter=0.0)

    def test_weno_coefficients_setup(self, simple_problem):
        """Test that WENO coefficients are properly initialized."""
        solver = HJBWenoSolver(simple_problem, weno_variant="weno5")

        # Check linear weights
        assert hasattr(solver, "d_plus")
        assert hasattr(solver, "d_minus")
        assert np.allclose(np.sum(solver.d_plus), 1.0)
        assert np.allclose(np.sum(solver.d_minus), 1.0)

        # Check reconstruction coefficients
        assert hasattr(solver, "c_plus")
        assert hasattr(solver, "c_minus")
        assert solver.c_plus.shape == (3, 3)
        assert solver.c_minus.shape == (3, 3)

    def test_smoothness_indicators_computation(self, simple_problem, test_values):
        """Test smoothness indicator computation for polynomial data."""
        solver = HJBWenoSolver(simple_problem, weno_variant="weno5")

        # For quadratic polynomial u = x^2, should have specific smoothness properties
        u_stencil = test_values[8:13]  # 5-point stencil
        beta = solver._compute_smoothness_indicators(u_stencil)

        # Should return 3 smoothness indicators
        assert len(beta) == 3
        assert all(beta >= 0)  # Smoothness indicators should be non-negative

        # For smooth quadratic data, middle stencil should be smoothest
        # (This is problem-dependent but generally true for polynomials)
        assert np.isfinite(beta).all()

    def test_tau_indicator_computation(self, simple_problem, test_values):
        """Test global smoothness indicator τ for WENO-Z."""
        solver = HJBWenoSolver(simple_problem, weno_variant="weno-z")

        u_stencil = test_values[8:13]  # 5-point stencil
        tau = solver._compute_tau_indicator(u_stencil)

        # τ should be non-negative and finite
        assert tau >= 0
        assert np.isfinite(tau)

    def test_weight_computation_variants(self, simple_problem, test_values):
        """Test weight computation for different WENO variants."""
        variants = ["weno5", "weno-z", "weno-m", "weno-js"]

        for variant in variants:
            solver = HJBWenoSolver(simple_problem, weno_variant=variant)

            w_plus, w_minus = solver._compute_weno_weights(test_values, 10)

            # Weights should sum to 1
            assert np.allclose(np.sum(w_plus), 1.0), f"Failed for {variant}"
            assert np.allclose(np.sum(w_minus), 1.0), f"Failed for {variant}"

            # Weights should be non-negative
            assert all(w_plus >= 0), f"Failed for {variant}"
            assert all(w_minus >= 0), f"Failed for {variant}"

            # Should have 3 weights each
            assert len(w_plus) == 3
            assert len(w_minus) == 3

    def test_weno_reconstruction(self, simple_problem, test_values):
        """Test WENO reconstruction produces reasonable results."""
        solver = HJBWenoSolver(simple_problem, weno_variant="weno5")

        # Test reconstruction at interior point
        u_left, u_right = solver._weno_reconstruction(test_values, 10)

        # Both should be finite
        assert np.isfinite(u_left)
        assert np.isfinite(u_right)

        # For smooth data, left and right values should be similar
        # but not necessarily identical due to stencil asymmetry
        assert abs(u_left - u_right) < 10.0  # Reasonable bound

    def test_hjb_step_execution(self, simple_problem):
        """Test that HJB time step executes without errors."""
        solver = HJBWenoSolver(simple_problem, weno_variant="weno5")

        # Create test data
        bounds = simple_problem.geometry.get_bounds()
        x = np.linspace(bounds[0][0], bounds[1][0], simple_problem.geometry.get_grid_shape()[0])
        u_current = np.sin(2 * np.pi * x)
        m_current = np.ones_like(u_current) / len(u_current)
        dt = 0.001

        # Should execute without error
        u_new = solver.solve_hjb_step(u_current, m_current, dt)

        # Result should be same size and finite
        assert u_new.shape == u_current.shape
        assert np.isfinite(u_new).all()

    def test_time_integration_methods(self, simple_problem):
        """Test different time integration methods."""
        methods = ["tvd_rk3", "explicit_euler"]

        for method in methods:
            solver = HJBWenoSolver(simple_problem, weno_variant="weno5", time_integration=method)

            bounds = simple_problem.geometry.get_bounds()
            x = np.linspace(bounds[0][0], bounds[1][0], simple_problem.geometry.get_grid_shape()[0])
            u_current = np.sin(2 * np.pi * x)
            m_current = np.ones_like(u_current) / len(u_current)
            dt = 0.001

            # Should execute without error
            u_new = solver.solve_hjb_step(u_current, m_current, dt)
            assert np.isfinite(u_new).all()

    def test_variant_info_retrieval(self, simple_problem):
        """Test that variant information is properly retrieved."""
        variants = ["weno5", "weno-z", "weno-m", "weno-js"]

        for variant in variants:
            solver = HJBWenoSolver(simple_problem, weno_variant=variant)
            info = solver.get_variant_info()

            # Should contain required keys
            required_keys = ["name", "description", "characteristics", "best_for"]
            for key in required_keys:
                assert key in info
                assert isinstance(info[key], str)
                assert len(info[key]) > 0

    def test_stability_time_step_computation(self, simple_problem):
        """Test stable time step computation."""
        solver = HJBWenoSolver(simple_problem, weno_variant="weno5")

        bounds = simple_problem.geometry.get_bounds()
        Nx_points = simple_problem.geometry.get_grid_shape()[0]
        x = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        u_test = np.sin(2 * np.pi * x)
        m_test = np.ones_like(u_test) / len(u_test)

        dt_stable = solver._compute_dt_stable_1d(u_test, m_test)

        # Should be positive and finite
        assert dt_stable > 0
        assert np.isfinite(dt_stable)

        # Should be reasonable for the problem
        assert dt_stable < 1.0  # Should be much smaller than problem time scale

    def test_boundary_handling(self, simple_problem):
        """Test that boundary points are handled correctly."""
        solver = HJBWenoSolver(simple_problem, weno_variant="weno5")

        # Create data with boundary features
        bounds = simple_problem.geometry.get_bounds()
        Nx_points = simple_problem.geometry.get_grid_shape()[0]
        x = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        u_boundary = np.exp(-10 * (x - 0.1) ** 2) + np.exp(-10 * (x - 0.9) ** 2)

        # Test reconstruction near boundaries
        # Should not raise errors
        u_left_0, u_right_0 = solver._weno_reconstruction(u_boundary, 2)
        u_left_end, u_right_end = solver._weno_reconstruction(u_boundary, len(u_boundary) - 3)

        assert np.isfinite([u_left_0, u_right_0, u_left_end, u_right_end]).all()

    def test_variant_performance_consistency(self, simple_problem):
        """Test that all variants produce consistent results for smooth problems."""
        variants = ["weno5", "weno-z", "weno-m", "weno-js"]

        # Use smooth initial condition
        bounds = simple_problem.geometry.get_bounds()
        Nx_points = simple_problem.geometry.get_grid_shape()[0]
        x = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        u_initial = np.sin(2 * np.pi * x)
        m_initial = np.ones_like(u_initial) / len(u_initial)
        dt = 0.001

        results = {}
        for variant in variants:
            solver = HJBWenoSolver(simple_problem, weno_variant=variant)
            u_result = solver.solve_hjb_step(u_initial, m_initial, dt)
            results[variant] = u_result

        # All results should be finite
        for variant, result in results.items():
            assert np.isfinite(result).all(), f"Non-finite result for {variant}"

        # Results should be reasonably close for smooth problems
        # (Different variants may have some variation, but should be in same ballpark)
        max_values = [np.max(np.abs(result)) for result in results.values()]
        assert max(max_values) / min(max_values) < 10.0  # Factor of 10 tolerance

    @pytest.mark.parametrize("variant", ["weno5", "weno-z", "weno-m", "weno-js"])
    def test_individual_variant_functionality(self, simple_problem, variant):
        """Test each WENO variant individually."""
        solver = HJBWenoSolver(simple_problem, weno_variant=variant)

        # Basic functionality test
        bounds = simple_problem.geometry.get_bounds()
        Nx_points = simple_problem.geometry.get_grid_shape()[0]
        x = np.linspace(bounds[0][0], bounds[1][0], Nx_points)
        u = np.sin(np.pi * x)
        m = np.ones_like(u) / len(u)
        dt = 0.001

        u_new = solver.solve_hjb_step(u, m, dt)

        assert u_new.shape == u.shape
        assert np.isfinite(u_new).all()

        # Variant info should be accessible
        info = solver.get_variant_info()
        # Normalize both strings by removing hyphens for comparison
        assert variant.replace("-", "").upper() in info["name"].replace("-", "").upper()


class TestWenoSolverIntegration:
    """Integration tests for WENO solver solve_hjb_system method."""

    @pytest.fixture
    def integration_problem(self) -> MFGProblem:
        """Create MFG problem for integration testing using modern geometry-first API."""
        domain = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[41], boundary_conditions=no_flux_bc(dimension=1))
        return MFGProblem(geometry=domain, T=1.0, Nt=30, sigma=0.1, components=_default_components())

    def test_solve_hjb_system_shape(self, integration_problem):
        """Test that solve_hjb_system returns correct shape."""
        solver = HJBWenoSolver(integration_problem, weno_variant="weno5")

        Nt = integration_problem.Nt + 1
        Nx = integration_problem.geometry.get_grid_shape()[0]  # Nx+1 grid points

        # Create inputs
        M_density = np.ones((Nt, Nx))
        U_final = np.zeros(Nx)
        U_prev = np.zeros((Nt, Nx))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert U_solution.shape == (Nt, Nx)
        assert np.all(np.isfinite(U_solution))

    def test_solve_hjb_system_final_condition(self, integration_problem):
        """Test that final condition is preserved."""
        solver = HJBWenoSolver(integration_problem, weno_variant="weno5")

        Nt = integration_problem.Nt + 1
        Nx = integration_problem.geometry.get_grid_shape()[0]  # Nx+1 grid points

        # Create inputs with specific final condition
        M_density = np.ones((Nt, Nx))
        bounds = integration_problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx)
        U_final = 0.5 * (x_coords - bounds[1][0]) ** 2
        U_prev = np.zeros((Nt, Nx))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Final time step should match final condition
        assert np.allclose(U_solution[-1, :], U_final, rtol=0.1)

    def test_solve_hjb_system_backward_propagation(self, integration_problem):
        """Test that solution propagates backward in time."""
        solver = HJBWenoSolver(integration_problem, weno_variant="weno5")

        Nt = integration_problem.Nt + 1
        Nx = integration_problem.geometry.get_grid_shape()[0]  # Nx+1 grid points

        # Create inputs
        M_density = np.ones((Nt, Nx))
        bounds = integration_problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx)
        U_final = x_coords**2  # Quadratic final condition
        U_prev = np.zeros((Nt, Nx))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Solution should propagate backward
        assert not np.allclose(U_solution[0, :], 0.0)

    def test_solve_hjb_system_with_density_variation(self, integration_problem):
        """Test solving with non-uniform density."""
        solver = HJBWenoSolver(integration_problem, weno_variant="weno5")

        Nt = integration_problem.Nt + 1
        Nx = integration_problem.geometry.get_grid_shape()[0]  # Nx+1 grid points

        # Create Gaussian density
        bounds = integration_problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx)
        m_profile = np.exp(-((x_coords - 0.5) ** 2) / (2 * 0.1**2))
        M_density = np.tile(m_profile, (Nt, 1))

        U_final = np.zeros(Nx)
        U_prev = np.zeros((Nt, Nx))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt, Nx)

    @pytest.mark.parametrize("variant", ["weno5", "weno-z", "weno-m", "weno-js"])
    def test_solve_hjb_system_all_variants(self, integration_problem, variant):
        """Test solve_hjb_system with all WENO variants."""
        solver = HJBWenoSolver(integration_problem, weno_variant=variant)

        Nt = integration_problem.Nt + 1
        Nx = integration_problem.geometry.get_grid_shape()[0]  # Nx+1 grid points

        M_density = np.ones((Nt, Nx))
        U_final = np.zeros(Nx)
        U_prev = np.zeros((Nt, Nx))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        assert U_solution.shape == (Nt, Nx)
        assert np.all(np.isfinite(U_solution))

    def test_solve_with_uniform_density(self, integration_problem):
        """Test solver with uniform density distribution."""
        solver = HJBWenoSolver(integration_problem, weno_variant="weno5")

        Nt = integration_problem.Nt + 1
        Nx = integration_problem.geometry.get_grid_shape()[0]  # Nx+1 grid points

        # Uniform density
        M_density = np.ones((Nt, Nx)) / Nx

        # Simple final condition
        bounds = integration_problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx)
        U_final = (x_coords - 0.5) ** 2

        U_prev = np.zeros((Nt, Nx))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Should produce valid solution
        assert np.all(np.isfinite(U_solution))
        assert U_solution.shape == (Nt, Nx)

    @pytest.mark.xfail(
        strict=True,
        reason="WENO HJB is spatially unstable for oscillatory terminal data (sin(2*pi*x), "
        "sigma=0.1): it amplifies high-frequency modes independent of dt -- see #1200. "
        "Pre-#1180 this passed only because the solver under-integrated (near-frozen at the "
        "bounded terminal) and never reached the blow-up. Remove the xfail when #1200 lands.",
    )
    def test_solution_finiteness(self, integration_problem):
        """Test that solution remains finite throughout (oscillatory terminal: #1200 instability)."""
        solver = HJBWenoSolver(integration_problem, weno_variant="weno5")

        Nt = integration_problem.Nt + 1
        Nx = integration_problem.geometry.get_grid_shape()[0]  # Nx+1 grid points

        M_density = np.ones((Nt, Nx)) * 0.5
        bounds = integration_problem.geometry.get_bounds()
        x_coords = np.linspace(bounds[0][0], bounds[1][0], Nx)
        U_final = np.sin(2 * np.pi * x_coords)
        U_prev = np.zeros((Nt, Nx))

        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # All values should be finite
        assert np.all(np.isfinite(U_solution))

    def test_different_cfl_numbers(self, integration_problem):
        """Test solver with different CFL numbers."""
        for cfl in [0.1, 0.3, 0.5]:
            solver = HJBWenoSolver(integration_problem, weno_variant="weno5", cfl_number=cfl)

            Nt = integration_problem.Nt + 1
            Nx = integration_problem.geometry.get_grid_shape()[0]  # Nx+1 grid points

            M_density = np.ones((Nt, Nx))
            U_final = np.zeros(Nx)
            U_prev = np.zeros((Nt, Nx))

            U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

            assert np.all(np.isfinite(U_solution))

    def test_solver_not_abstract(self, integration_problem):
        """Test that HJBWenoSolver can be instantiated and used."""
        import inspect

        # Should not raise TypeError about abstract methods
        solver = HJBWenoSolver(integration_problem, weno_variant="weno5")
        assert isinstance(solver, HJBWenoSolver)

        # Should not have abstract methods
        assert not inspect.isabstract(HJBWenoSolver)


class TestWenoTimeSubstepping:
    """Issue #1180: each backward interval must sub-step to cover the full physical ``dt``,
    not advance only one CFL/diffusion-stable ``dt_stable`` (which near-froze the value
    function at the terminal condition in the common diffusion-limited regime)."""

    @staticmethod
    def _problem(Nx=51, T=1.0, Nt=20, sigma=0.3):
        ham = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        )
        comps = MFGComponents(
            m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
            u_terminal=lambda x: 0.5 * (x - 0.5) ** 2,
            hamiltonian=ham,
        )
        dom = TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=no_flux_bc(dimension=1)
        )
        return MFGProblem(geometry=dom, T=T, Nt=Nt, sigma=sigma, components=comps)

    @pytest.mark.slow
    def test_integrates_full_horizon_not_one_dt_stable(self):
        """Diffusion-limited (sigma=0.3, dt/dt_stable ~ 45): the backward sweep must transport
        the value function over the full horizon, matching a full-horizon substepped reference
        built from the same kernel. Pre-fix the sweep advanced only ~2% of T (moved ~0.044 vs
        reference ~0.295) because each interval took a single dt_stable step."""
        prob = self._problem()
        solver = HJBWenoSolver(problem=prob, weno_variant="weno5")
        n_time = prob.Nt + 1
        Nx = solver.num_grid_points_x
        x = np.linspace(0.0, 1.0, Nx)
        U_T = 0.5 * (x - 0.5) ** 2
        m_row = np.ones(Nx) / Nx
        M = np.tile(m_row, (n_time, 1))
        U_prev = np.tile(U_T, (n_time, 1))

        U = solver._solve_hjb_system_1d(M, U_T, U_prev)
        moved = np.linalg.norm(U[0] - U[-1])

        # Reference: the same kernel substepped continuously over the full horizon T.
        u = U_T.copy()
        t = 0.0
        guard = 0
        while t < prob.T - 1e-12 and guard < 100_000:
            ds = min(solver._compute_dt_stable_1d(u, m_row), prob.T - t)
            u = solver.solve_hjb_step(u, m_row, ds)
            t += ds
            guard += 1
        moved_ref = np.linalg.norm(u - U_T)

        assert moved_ref > 1e-3, "reference horizon transport is trivial; test not exercising the path"
        assert abs(moved - moved_ref) / moved_ref < 0.15, (
            f"value function under-propagated: sweep moved={moved:.4e} vs full-horizon ref={moved_ref:.4e}"
        )

    def test_single_step_when_dt_below_stable_is_byte_identical(self):
        """Happy path: when dt <= dt_stable the substep helper does exactly one step of size dt,
        byte-identical to the pre-#1180 single-step code (dt_stable_fn returns a huge value)."""
        prob = self._problem(Nx=11, T=0.01, Nt=5, sigma=0.05)
        solver = HJBWenoSolver(problem=prob, weno_variant="weno5")
        x = np.linspace(0.0, 1.0, 11)
        u = np.cos(np.pi * x)
        m = np.ones(11) / 11
        dt = 0.002
        one_step = solver.solve_hjb_step(u, m, dt)
        via_helper = solver._advance_full_interval(u, m, dt, lambda uu, mm: 1e9, solver.solve_hjb_step)
        np.testing.assert_array_equal(via_helper, one_step)

    def test_fails_loud_at_max_substeps(self):
        """Fail-fast contract: if the CFL/diffusion limit cannot cover dt within max_substeps,
        raise rather than silently truncating the interval."""
        prob = self._problem(Nx=11, T=0.01, Nt=5, sigma=0.05)
        solver = HJBWenoSolver(problem=prob, weno_variant="weno5")
        solver.max_substeps = 3
        u = np.cos(np.pi * np.linspace(0.0, 1.0, 11))
        m = np.ones(11) / 11
        with pytest.raises(ValueError, match="max_substeps"):
            # 3 substeps of 0.01 cover 0.03 << dt=1.0 -> uncovered -> raise
            solver._advance_full_interval(u, m, 1.0, lambda uu, mm: 0.01, lambda uu, mm, dd: uu)


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
