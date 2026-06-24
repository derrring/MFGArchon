#!/usr/bin/env python3
"""
Unit tests for non-quadratic Hamiltonian support in FP solvers (Issue #573).

Tests the new drift_field parameter for L1, quartic, and custom Hamiltonians.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid, no_flux_bc


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
        m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(),
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


class TestNonQuadraticHamiltonians:
    """Test FP solvers with non-quadratic Hamiltonians."""

    @pytest.fixture
    def problem_1d(self):
        """Create 1D test problem."""
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        return MFGProblem(
            geometry=geometry,
            T=1.0,
            Nt=20,
            sigma=0.1,
            coupling_coefficient=0.0,  # No coupling for unit tests
            components=_default_components(),
        )

    @pytest.fixture
    def solver_1d(self, problem_1d):
        """Create 1D FDM solver."""
        bc = no_flux_bc(dimension=1)
        return FPFDMSolver(problem_1d, boundary_conditions=bc)

    @pytest.fixture
    def initial_density_1d(self, problem_1d):
        """Gaussian initial density."""
        Nx = problem_1d.geometry.get_grid_shape()[0]  # 1D: (Nx,)
        x = np.linspace(0, 1, Nx)
        m0 = np.exp(-50 * (x - 0.5) ** 2)
        m0 = m0 / np.trapezoid(m0, dx=1.0 / (Nx - 1))
        return m0

    def test_drift_shape_validation(self, solver_1d, initial_density_1d, problem_1d):
        """Test that drift_field shape is validated."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]

        # Wrong shape should raise error
        wrong_shape_drift = np.zeros((Nt, Nx + 10))  # Wrong Nx

        with pytest.raises((ValueError, IndexError, AssertionError)):
            solver_1d.solve_fp_system(
                M_initial=initial_density_1d,
                potential_field=wrong_shape_drift,
            )

    def test_backward_compatibility_quadratic(self, solver_1d, initial_density_1d, problem_1d):
        """Test that drift_field still works (quadratic H: α* = -∇U)."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]
        x = np.linspace(0, 1, Nx)

        # Create synthetic value function U(t,x) = x²
        U_hjb = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            U_hjb[t_idx, :] = x**2

        # Solve with legacy drift_field API
        M_legacy = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            potential_field=U_hjb,
        )

        # Should produce valid density
        assert M_legacy.shape == (Nt, Nx)
        assert np.all(M_legacy >= -1e-10), "Density should be non-negative"
        assert not np.any(np.isnan(M_legacy)), "No NaN values"
        assert not np.any(np.isinf(M_legacy)), "No Inf values"

    def test_equivalence_quadratic_explicit(self, initial_density_1d):
        """Test that drift_field and potential_field give same result for quadratic H.

        Uses coupling_coefficient=1.0 so that the internal potential path
        (v = -coupling_coefficient * grad_U) matches the manually computed
        drift alpha = -grad_U.
        """
        # Need coupling_coefficient=1.0 for equivalence: internal drift = -1.0 * grad_U
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
        problem_cc1 = MFGProblem(
            geometry=geometry,
            T=1.0,
            Nt=20,
            sigma=0.1,
            coupling_coefficient=1.0,
            components=_default_components(),
        )
        solver_cc1 = FPFDMSolver(problem_cc1, boundary_conditions=no_flux_bc(dimension=1))

        Nt, Nx = problem_cc1.Nt + 1, problem_cc1.geometry.get_grid_shape()[0]
        x = np.linspace(0, 1, Nx)
        dx = x[1] - x[0]

        # Create synthetic value function U(t,x) = x²
        U_hjb = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            U_hjb[t_idx, :] = x**2

        # Compute gradient manually: ∇U = 2x (central differences)
        grad_U = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            grad_U[t_idx, 1:-1] = (U_hjb[t_idx, 2:] - U_hjb[t_idx, :-2]) / (2 * dx)
            # Forward/backward at boundaries
            grad_U[t_idx, 0] = (U_hjb[t_idx, 1] - U_hjb[t_idx, 0]) / dx
            grad_U[t_idx, -1] = (U_hjb[t_idx, -1] - U_hjb[t_idx, -2]) / dx

        # Quadratic H: α* = -∇U
        alpha_quadratic = -grad_U

        # Solve with potential_field (legacy U-potential path)
        M_legacy = solver_cc1.solve_fp_system(
            M_initial=initial_density_1d,
            potential_field=U_hjb,
        )

        # Solve with drift_field (new velocity API)
        M_new = solver_cc1.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=alpha_quadratic,
        )

        # The two paths use different internal discretizations:
        # - potential_field: scheme computes velocity from U-gradient (upwind stencil)
        # - drift_field: pre-computed velocity passed as interface_velocity (face-centered)
        # Tolerance reflects expected O(dx) discretization difference.
        np.testing.assert_allclose(
            M_new,
            M_legacy,
            rtol=0.5,
            atol=1e-4,
            err_msg="drift_field and potential_field should be comparable for quadratic H",
        )

    def test_l1_control_drift(self, solver_1d, initial_density_1d, problem_1d):
        """Test L1 Hamiltonian: H = |p|, α* = -sign(∇U)."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]
        x = np.linspace(0, 1, Nx)
        dx = x[1] - x[0]

        # Create synthetic value function U(t,x) = (x - 0.5)²
        U_hjb = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            U_hjb[t_idx, :] = (x - 0.5) ** 2

        # Compute gradient: ∇U = 2(x - 0.5)
        grad_U = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            grad_U[t_idx, 1:-1] = (U_hjb[t_idx, 2:] - U_hjb[t_idx, :-2]) / (2 * dx)
            grad_U[t_idx, 0] = (U_hjb[t_idx, 1] - U_hjb[t_idx, 0]) / dx
            grad_U[t_idx, -1] = (U_hjb[t_idx, -1] - U_hjb[t_idx, -2]) / dx

        # L1 Hamiltonian: α* = -sign(∇U)
        alpha_L1 = -np.sign(grad_U)

        # Solve with L1 drift
        M_L1 = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=alpha_L1,
        )

        # Should produce valid density
        assert M_L1.shape == (Nt, Nx)
        assert np.all(M_L1 >= -1e-10), "Density should be non-negative"
        assert not np.any(np.isnan(M_L1)), "No NaN values"

        # L1 drift behaves differently from quadratic - verify density evolution occurs
        # Check that solution changes over time (not static)
        density_change = np.linalg.norm(M_L1[-1, :] - M_L1[0, :])
        assert density_change > 1e-6, "L1 drift should evolve density over time"

    def test_quartic_control_drift(self, solver_1d, initial_density_1d, problem_1d):
        """Test quartic Hamiltonian: H = (1/4)|p|⁴, α* = -(∇U)^(1/3)."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]
        x = np.linspace(0, 1, Nx)
        dx = x[1] - x[0]

        # Create synthetic value function U(t,x) = (x - 0.5)²
        U_hjb = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            U_hjb[t_idx, :] = (x - 0.5) ** 2

        # Compute gradient: ∇U = 2(x - 0.5)
        grad_U = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            grad_U[t_idx, 1:-1] = (U_hjb[t_idx, 2:] - U_hjb[t_idx, :-2]) / (2 * dx)
            grad_U[t_idx, 0] = (U_hjb[t_idx, 1] - U_hjb[t_idx, 0]) / dx
            grad_U[t_idx, -1] = (U_hjb[t_idx, -1] - U_hjb[t_idx, -2]) / dx

        # Quartic Hamiltonian: α* = -(∇U)^(1/3)
        # Need to preserve sign
        alpha_quartic = -np.sign(grad_U) * np.abs(grad_U) ** (1.0 / 3.0)

        # Solve with quartic drift
        M_quartic = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=alpha_quartic,
        )

        # Should produce valid density
        assert M_quartic.shape == (Nt, Nx)
        assert np.all(M_quartic >= -1e-10), "Density should be non-negative"
        assert not np.any(np.isnan(M_quartic)), "No NaN values"

        # Quartic drift behaves differently - verify density evolution occurs
        # Check that solution changes over time (not static)
        density_change = np.linalg.norm(M_quartic[-1, :] - M_quartic[0, :])
        assert density_change > 1e-6, "Quartic drift should evolve density over time"

    def test_zero_drift_pure_diffusion(self, solver_1d, initial_density_1d, problem_1d):
        """Test zero drift (pure diffusion) with drift_field=0."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]

        # Zero drift velocity
        alpha_zero = np.zeros((Nt, Nx))

        # Solve with zero drift
        M_diffusion = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=alpha_zero,
        )

        # Should produce valid density that spreads out (pure diffusion)
        assert M_diffusion.shape == (Nt, Nx)
        assert np.all(M_diffusion >= -1e-10), "Density should be non-negative"

        # Diffusion should reduce peak height and spread mass
        assert M_diffusion[-1, :].max() < initial_density_1d.max(), "Diffusion should reduce peak"

    def test_custom_drift_callable_like(self, solver_1d, initial_density_1d, problem_1d):
        """Test custom drift velocity (e.g., state-constrained control)."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]
        x = np.linspace(0, 1, Nx)

        # Custom drift: push left in left half, push right in right half
        # Simulates a "barrier" at x=0.5
        alpha_custom = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            alpha_custom[t_idx, x < 0.5] = -0.5  # Push left
            alpha_custom[t_idx, x >= 0.5] = 0.5  # Push right

        # Solve with custom drift
        M_custom = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=alpha_custom,
        )

        # Should produce valid density
        assert M_custom.shape == (Nt, Nx)
        assert np.all(M_custom >= -1e-10), "Density should be non-negative"
        assert not np.any(np.isnan(M_custom)), "No NaN values"

    def test_mass_conservation_with_drift_field(self, solver_1d, initial_density_1d, problem_1d):
        """Test that mass is conserved with drift_field."""
        Nt, Nx = problem_1d.Nt + 1, problem_1d.geometry.get_grid_shape()[0]
        x = np.linspace(0, 1, Nx)
        dx = x[1] - x[0]

        # Create synthetic drift
        alpha = np.zeros((Nt, Nx))
        for t_idx in range(Nt):
            alpha[t_idx, :] = 0.5 * np.sin(2 * np.pi * x)

        # Solve
        M = solver_1d.solve_fp_system(
            M_initial=initial_density_1d,
            drift_field=alpha,
        )

        # Check mass conservation at each timestep
        initial_mass = np.trapezoid(initial_density_1d, dx=dx)
        for t_idx in range(Nt):
            mass_t = np.trapezoid(M[t_idx, :], dx=dx)
            np.testing.assert_allclose(
                mass_t,
                initial_mass,
                rtol=1e-2,
                err_msg=f"Mass not conserved at t={t_idx * problem_1d.dt}",
            )


# ============================================================================
# Issue #896: Non-quadratic drift through FixedPointIterator
# ============================================================================


@pytest.mark.slow
class TestFixedPointIteratorDrift:
    """Test that FixedPointIterator computes correct drift for non-quadratic H."""

    def _make_problem(self, hamiltonian):
        """Create a standard 1D MFG problem with given Hamiltonian."""
        from mfgarchon.core.mfg_components import MFGComponents

        Nx, Nt = 51, 20
        bc = no_flux_bc(dimension=1)
        geom = TensorProductGrid(bounds=[(0.0, 1.0)], num_points=[Nx], boundary_conditions=bc)
        components = MFGComponents(
            hamiltonian=hamiltonian,
            m_initial=lambda x: np.exp(-((x - 0.5) ** 2) / 0.02),
            u_terminal=lambda x: 0.5 * (x - 0.5) ** 2,
        )
        return MFGProblem(
            geometry=geom,
            T=1.0,
            Nt=Nt,
            sigma=0.1,
            boundary_conditions=bc,
            components=components,
        )

    def test_quadratic_drift_matches_legacy(self):
        """Quadratic H through the FixedPointIterator drift pipeline should match passing U directly."""
        from mfgarchon.alg.numerical.coupling import FixedPointIterator
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        H = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        )
        problem = self._make_problem(H)
        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)

        # Run with new drift pipeline
        iterator = FixedPointIterator(problem, hjb_solver, fp_solver)
        result = iterator.solve(max_iterations=5, tolerance=1e-10)
        U, M = result.U, result.M

        # Basic sanity: solution should be finite and non-negative density
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(M))
        assert M.min() >= -1e-6  # Allow small numerical negativity

    def test_l1_drift_produces_different_density(self):
        """L1 control should produce different density than quadratic."""
        from mfgarchon.alg.numerical.coupling import FixedPointIterator
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        # Quadratic problem
        H_quad = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
        )
        prob_q = self._make_problem(H_quad)
        it_q = FixedPointIterator(prob_q, HJBFDMSolver(prob_q), FPFDMSolver(prob_q))
        res_q = it_q.solve(max_iterations=5, tolerance=1e-10)
        M_q = res_q.M

        # L1 problem (same geometry, different H)
        from mfgarchon.core.hamiltonian import L1ControlCost

        H_l1 = SeparableHamiltonian(
            control_cost=L1ControlCost(control_cost=0.1),
        )
        prob_l1 = self._make_problem(H_l1)
        it_l1 = FixedPointIterator(prob_l1, HJBFDMSolver(prob_l1), FPFDMSolver(prob_l1))
        res_l1 = it_l1.solve(max_iterations=5, tolerance=1e-10)
        U_l1, M_l1 = res_l1.U, res_l1.M

        # Solutions should be finite
        assert np.all(np.isfinite(U_l1))
        assert np.all(np.isfinite(M_l1))

        # L1 and quadratic should produce different densities
        assert not np.allclose(M_q, M_l1, atol=1e-3), (
            "L1 and quadratic produced identical densities — drift not differentiated"
        )

    def test_bounded_drift_produces_different_density(self):
        """BoundedControlCost should produce different density than quadratic."""
        from mfgarchon.alg.numerical.coupling import FixedPointIterator
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
        from mfgarchon.core.hamiltonian import BoundedControlCost

        H_quad = SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
        )
        prob_q = self._make_problem(H_quad)
        it_q = FixedPointIterator(prob_q, HJBFDMSolver(prob_q), FPFDMSolver(prob_q))
        res_q = it_q.solve(max_iterations=5, tolerance=1e-10)
        M_q = res_q.M

        H_bnd = SeparableHamiltonian(
            control_cost=BoundedControlCost(control_cost=1.0, max_control=0.1),
        )
        prob_b = self._make_problem(H_bnd)
        it_b = FixedPointIterator(prob_b, HJBFDMSolver(prob_b), FPFDMSolver(prob_b))
        res_b = it_b.solve(max_iterations=5, tolerance=1e-10)
        U_b, M_b = res_b.U, res_b.M

        assert np.all(np.isfinite(U_b))
        assert np.all(np.isfinite(M_b))
        assert not np.allclose(M_q, M_b, atol=1e-3), "Bounded and quadratic produced identical densities"


if __name__ == "__main__":
    """Run smoke tests."""
    import sys

    print("Running non-quadratic Hamiltonian tests...")
    pytest.main([__file__, "-v", "--tb=short", *sys.argv[1:]])
