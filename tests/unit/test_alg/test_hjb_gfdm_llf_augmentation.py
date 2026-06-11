"""
Pinning tests for LLF (Local Lax-Friedrichs) augmented diffusion in HJBGFDMSolver.

Issue #1059: paper P2 branch of thm:discrete_comparison — per-node artificial viscosity
nu_i that restores the discrete comparison principle at high-Pe nodes.

PINNING CONTRACT
----------------
Pre-fix (before this PR): ``HJBGFDMSolver.__init__`` has no ``llf_augmentation``
parameter; constructing with it raises ``TypeError: unexpected keyword argument``.

Post-fix: the parameter exists; the solver stores ``_llf_sigma_eff`` (per-node effective
sigma), and ``_get_sigma_value(i)`` returns ``sigma_eff_i >= sigma`` for all i when LLF is
active.  With LLF OFF the result is byte-identical to the baseline (no sigma override).
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _hamiltonian():
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _components():
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: 0.0,
        hamiltonian=_hamiltonian(),
    )


@pytest.fixture
def problem_and_pts():
    """1D MFG problem + 21 uniform collocation points."""
    sigma = 0.5  # small sigma → high Pe (motivating regime for LLF)
    domain = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[21],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    problem = MFGProblem(geometry=domain, T=1.0, Nt=21, sigma=sigma, components=_components())
    bounds = problem.geometry.get_bounds()
    (Nx,) = problem.geometry.get_grid_shape()
    pts = np.linspace(bounds[0][0], bounds[1][0], Nx).reshape(-1, 1)
    return problem, pts


# ---------------------------------------------------------------------------
# PINNING TEST 1 — core: option accepted + sigma_eff stored correctly
# ---------------------------------------------------------------------------


class TestLLFAugmentationPinning:
    """Core pinning tests — fail pre-fix, pass post-fix."""

    def test_llf_parameter_accepted(self, problem_and_pts):
        """PINNING: llf_augmentation parameter accepted without TypeError.

        Pre-fix: TypeError: __init__() got an unexpected keyword argument 'llf_augmentation'.
        Post-fix: solver constructs cleanly.
        """
        problem, pts = problem_and_pts
        # This raises TypeError on the unpatched code.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solver = HJBGFDMSolver(
                problem,
                pts,
                monotonicity_scheme="none",
                llf_augmentation=True,
                llf_cone_constant=0.5,
                llf_l_H=10.0,  # large l_H guarantees nu_i > 0 at every node
            )
        assert solver.llf_augmentation is True

    def test_llf_sigma_eff_stored(self, problem_and_pts):
        """PINNING: _llf_sigma_eff is a shape-(n_points,) float array when LLF is on."""
        problem, pts = problem_and_pts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solver = HJBGFDMSolver(
                problem,
                pts,
                monotonicity_scheme="none",
                llf_augmentation=True,
                llf_l_H=10.0,
            )
        assert solver._llf_sigma_eff is not None
        assert solver._llf_sigma_eff.shape == (solver.n_points,)
        assert solver._llf_sigma_eff.dtype == np.float64

    def test_llf_sigma_eff_ge_base(self, problem_and_pts):
        """PINNING: sigma_eff_i >= sigma everywhere (LLF only adds diffusion)."""
        problem, pts = problem_and_pts
        sigma_base = problem.sigma
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solver = HJBGFDMSolver(
                problem,
                pts,
                monotonicity_scheme="none",
                llf_augmentation=True,
                llf_l_H=10.0,
            )
        np.testing.assert_array_less(
            sigma_base - 1e-12,
            solver._llf_sigma_eff,
            err_msg="sigma_eff_i must be >= sigma (LLF only stabilises, never de-stabilises)",
        )

    def test_llf_at_least_one_node_augmented(self, problem_and_pts):
        """PINNING: with l_H=10 and delta=0.1, nu_i > 0 at every node."""
        problem, pts = problem_and_pts
        # nu_i = max(0, C*l_H*delta - sigma^2/2) = max(0, 0.5*10*0.1 - 0.25/2)
        #       = max(0, 0.5 - 0.125) = 0.375 > 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solver = HJBGFDMSolver(
                problem,
                pts,
                monotonicity_scheme="none",
                llf_augmentation=True,
                llf_l_H=10.0,
            )
        assert np.all(solver._llf_sigma_eff > problem.sigma + 1e-12), (
            "Expected all nodes augmented for l_H=10, delta=0.1, sigma=0.5, C=0.5"
        )

    def test_llf_get_sigma_value_returns_eff(self, problem_and_pts):
        """PINNING: _get_sigma_value(i) returns sigma_eff_i when LLF is on."""
        problem, pts = problem_and_pts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solver = HJBGFDMSolver(
                problem,
                pts,
                monotonicity_scheme="none",
                llf_augmentation=True,
                llf_l_H=10.0,
            )
        for i in range(min(5, solver.n_points)):
            got = solver._get_sigma_value(i)
            expected = float(solver._llf_sigma_eff[i])
            assert abs(got - expected) < 1e-12, f"_get_sigma_value({i}) = {got} != sigma_eff[{i}] = {expected}"

    def test_llf_off_get_sigma_value_unchanged(self, problem_and_pts):
        """PINNING: LLF OFF → _get_sigma_value(i) returns base problem sigma."""
        problem, pts = problem_and_pts
        sigma_base = float(problem.sigma)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solver = HJBGFDMSolver(problem, pts, monotonicity_scheme="none")
        for i in range(min(5, solver.n_points)):
            got = solver._get_sigma_value(i)
            assert abs(got - sigma_base) < 1e-12, f"LLF OFF: _get_sigma_value({i}) = {got} != sigma = {sigma_base}"

    def test_llf_off_no_sigma_eff(self, problem_and_pts):
        """PINNING: LLF OFF → _llf_sigma_eff is None (zero overhead)."""
        problem, pts = problem_and_pts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solver = HJBGFDMSolver(problem, pts, monotonicity_scheme="none")
        assert solver._llf_sigma_eff is None


# ---------------------------------------------------------------------------
# Fail-loud validation
# ---------------------------------------------------------------------------


class TestLLFValidation:
    """Fail-loud validation tests (fail-fast per CLAUDE.md)."""

    def test_missing_l_H_raises(self, problem_and_pts):
        """llf_augmentation=True without llf_l_H raises ValueError (fail-loud)."""
        problem, pts = problem_and_pts
        # Suppress unrelated UserWarning from no monotonicity_scheme; ValueError is what
        # we're testing, and it fires before the warning in the constructor path.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="llf_l_H"):
                HJBGFDMSolver(problem, pts, monotonicity_scheme="none", llf_augmentation=True)

    def test_negative_l_H_raises(self, problem_and_pts):
        """Negative l_H values raise ValueError."""
        problem, pts = problem_and_pts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="non-negative"):
                HJBGFDMSolver(
                    problem,
                    pts,
                    monotonicity_scheme="none",
                    llf_augmentation=True,
                    llf_l_H=-1.0,
                )


# ---------------------------------------------------------------------------
# Numerical correctness of nu_i computation
# ---------------------------------------------------------------------------


class TestLLFNumerics:
    """Numerical correctness of sigma_eff_i computation."""

    def test_zero_l_H_gives_base_sigma(self, problem_and_pts):
        """l_H = 0 → nu_i = 0 → sigma_eff_i = sigma (no augmentation at any node)."""
        problem, pts = problem_and_pts
        sigma_base = float(problem.sigma)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solver = HJBGFDMSolver(
                problem,
                pts,
                monotonicity_scheme="none",
                llf_augmentation=True,
                llf_l_H=0.0,
            )
        np.testing.assert_allclose(
            solver._llf_sigma_eff,
            sigma_base,
            rtol=1e-12,
            err_msg="l_H=0 → nu_i=0 → sigma_eff_i must equal sigma exactly",
        )

    def test_sigma_eff_formula_scalar_l_H(self, problem_and_pts):
        """Verify sigma_eff_i matches the analytic formula for scalar l_H."""
        problem, pts = problem_and_pts
        sigma = float(problem.sigma)
        C = 0.5
        l_H = 5.0
        delta = 0.1  # default

        # Expected: nu_i = max(0, C*l_H*delta - sigma^2/2), same at every node
        D_base = 0.5 * sigma**2
        nu_expected = max(0.0, C * l_H * delta - D_base)
        sigma_eff_expected = np.sqrt(sigma**2 + 2.0 * nu_expected)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solver = HJBGFDMSolver(
                problem,
                pts,
                monotonicity_scheme="none",
                llf_augmentation=True,
                llf_cone_constant=C,
                llf_l_H=l_H,
            )
        np.testing.assert_allclose(
            solver._llf_sigma_eff,
            sigma_eff_expected,
            rtol=1e-12,
            err_msg="sigma_eff_i must match analytic formula",
        )

    def test_sigma_eff_formula_per_node_l_H(self, problem_and_pts):
        """sigma_eff_i computed correctly when l_H is a per-node array."""
        problem, pts = problem_and_pts
        sigma = float(problem.sigma)
        C = 0.5
        delta = 0.1
        n = pts.shape[0]

        # Varying l_H: first half high, second half zero
        l_H_arr = np.zeros(n)
        l_H_arr[: n // 2] = 20.0  # augmented nodes
        l_H_arr[n // 2 :] = 0.0  # unaugmented nodes

        D_base = 0.5 * sigma**2
        nu_expected = np.maximum(0.0, C * l_H_arr * delta - D_base)
        sigma_eff_expected = np.sqrt(sigma**2 + 2.0 * nu_expected)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solver = HJBGFDMSolver(
                problem,
                pts,
                monotonicity_scheme="none",
                llf_augmentation=True,
                llf_cone_constant=C,
                llf_l_H=l_H_arr,
            )
        np.testing.assert_allclose(
            solver._llf_sigma_eff,
            sigma_eff_expected,
            rtol=1e-12,
            err_msg="Per-node sigma_eff_i must match element-wise formula",
        )

    def test_cone_constant_effect(self, problem_and_pts):
        """Larger C → larger nu_i → larger sigma_eff_i."""
        problem, pts = problem_and_pts
        l_H = 8.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solver_c05 = HJBGFDMSolver(
                problem,
                pts,
                monotonicity_scheme="none",
                llf_augmentation=True,
                llf_cone_constant=0.5,
                llf_l_H=l_H,
            )
            solver_c10 = HJBGFDMSolver(
                problem,
                pts,
                monotonicity_scheme="none",
                llf_augmentation=True,
                llf_cone_constant=1.0,
                llf_l_H=l_H,
            )
        # C=1.0 → larger nu_i → larger sigma_eff
        assert np.all(solver_c10._llf_sigma_eff >= solver_c05._llf_sigma_eff - 1e-12)
        assert np.any(solver_c10._llf_sigma_eff > solver_c05._llf_sigma_eff + 1e-12)


# ---------------------------------------------------------------------------
# Operator-level test: Jacobian diagonal differs in stabilising direction
# ---------------------------------------------------------------------------


class TestLLFJacobianEffect:
    """Verify that LLF ON changes the assembled Jacobian in the expected direction."""

    def test_jacobian_diffusion_diagonal_larger_with_llf(self, problem_and_pts):
        """LLF ON → Jacobian diagonal is larger at interior nodes.

        The (i, i) entry of the vectorized Jacobian contains:
            (1/dt) - D_i * L_ii + (dH/dp) * G_ii
        The center Laplacian weight L_ii = -sum_{j!=i}(L_ij) < 0 (M-matrix: off-diagonal
        weights L_ij >= 0, so center is their negated sum).  Therefore:
            -D_i * L_ii = D_i * |L_ii| > 0   (positive contribution)

        With LLF, D_i_eff >= D_i, so -D_i_eff * L_ii >= -D_i * L_ii, meaning the
        diagonal is (algebraically) LARGER when LLF is active.  We verify this at
        interior nodes with zero gradient (no advection term).
        """
        problem, pts = problem_and_pts
        n = pts.shape[0]
        grad_u_zero = np.zeros((n, 1))  # zero gradient → no advection term

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            solver_off = HJBGFDMSolver(problem, pts, monotonicity_scheme="none")
            solver_on = HJBGFDMSolver(
                problem,
                pts,
                monotonicity_scheme="none",
                llf_augmentation=True,
                llf_l_H=10.0,
            )

        # Build differentiation matrices
        solver_off._build_differentiation_matrices()
        solver_on._build_differentiation_matrices()

        # Jacobian at zero gradient (only time + diffusion terms matter)
        J_off = solver_off._compute_hjb_jacobian_vectorized(grad_u_zero)
        J_on = solver_on._compute_hjb_jacobian_vectorized(grad_u_zero)

        diag_off = J_off.diagonal()
        diag_on = J_on.diagonal()

        # LLF adds diffusion → diagonal increases (D_i_eff > D_i, center lap weight < 0,
        # so -D_i_eff * L_ii > -D_i * L_ii).
        # Only interior rows are affected by diffusion coefficient change.
        interior = solver_on.interior_indices
        assert np.all(diag_on[interior] >= diag_off[interior] - 1e-12), (
            "LLF ON should produce Jacobian diagonal >= LLF OFF diagonal at interior nodes "
            "(larger D_i_eff, negative center Laplacian weight → larger positive contribution)"
        )
        # At least one interior node must differ strictly
        assert np.any(diag_on[interior] > diag_off[interior] + 1e-12), (
            "Expected at least one interior node with strictly larger diagonal "
            "when LLF is active with l_H=10 (nu_i > 0 at every node)"
        )
