"""The norm `FPSLSolver`'s density converges in (#2468; user ruling 2026-10-13: document now, #2645 for the scheme).

At σ = 0 the scheme converges in W₁ and L¹, which is what its theory gives (Carlini–Silva, arXiv:1708.02042, Theorem 4.1),
and not pointwise at a node where the drift changes sign: the linear deposit gives that node twice the continuity
equation's inflow. With σ > 0 the excess is diffused and the density converges pointwise. Each test below is an external
oracle (an exact law) or, for the stagnation node, a labelled pin of the documented behaviour.
"""

from __future__ import annotations

import itertools
import warnings

import pytest

import numpy as np

from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

C, T = 0.5, 1.0  # the drift -C (x - 1/2), from U = C (x - 1/2)^2 / 2: compressive, stagnating at x = 1/2


def _solve(nx: int, nt: int, horizon: float, sigma: float, m0: np.ndarray) -> np.ndarray:
    """``FPSLSolver`` (linear splat) on [0, 1] with no-flux walls, from ``m0`` under the drift -C (x - 1/2); M(T)."""
    x = np.linspace(0.0, 1.0, nx)
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[nx], boundary_conditions=no_flux_bc(dimension=1))
    components = MFGComponents(
        hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0)),
        m_initial=lambda y: 1.0,
        u_terminal=lambda y: 0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = MFGProblem(geometry=grid, components=components, T=horizon, Nt=nt, volatility=sigma)
    potential = np.tile(C * (x - 0.5) ** 2 / 2, (nt + 1, 1))
    return np.asarray(FPSLSolver(problem).solve_fp_system(m0, potential_field=potential))


def _w1(x: np.ndarray, f: np.ndarray, g: np.ndarray) -> float:
    """W₁ on the line: the L¹ distance of the two cumulative distributions (trapezoid)."""
    cumulative = lambda h: np.concatenate([[0.0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(x))])  # noqa: E731
    return float(np.trapezoid(np.abs(cumulative(f) - cumulative(g)), x))


def _issue_fixture(nx: int):
    """#2468's fixture: m0 ∝ exp(-10 (x - 1/2)²) on [0, 1], zero outside. Its exact law at T is the pushforward
    m0(1/2 + (x - 1/2) e^{CT}) e^{CT}, whose support is |x - 1/2| <= e^{-CT}/2."""
    x = np.linspace(0.0, 1.0, nx)
    z = np.trapezoid(np.exp(-10 * (x - 0.5) ** 2), x)
    y = 0.5 + (x - 0.5) * np.exp(C * T)
    exact = np.where(np.abs(y - 0.5) <= 0.5, np.exp(-10 * (y - 0.5) ** 2) * np.exp(C * T) / z, 0.0)
    return x, np.exp(-10 * (x - 0.5) ** 2) / z, exact


@pytest.fixture(scope="module")
def issue_runs():
    """#2468's fixture at nx = 51, 101, 201, dt = h: (x, M(T), exact) per level."""
    runs = []
    for nx in (51, 101, 201):
        x, m0, exact = _issue_fixture(nx)
        runs.append((x, _solve(nx, nx - 1, T, 0.0, m0)[-1], exact))
    return runs


def test_at_sigma_zero_the_density_converges_in_w1_and_l1(issue_runs):
    """W₁ and L¹ fall under refinement along dt = h, as Theorem 4.1's convergence in C([0,T]; P₁) allows. Measured at
    acdae412: W₁ 3.86e-3, 1.85e-3, 9.07e-4 (orders 1.06, 1.03) and L¹ 6.78e-2, 3.87e-2, 2.24e-2 (orders 0.81, 0.79)."""
    w1 = [_w1(x, m, exact) for x, m, exact in issue_runs]
    l1 = [float(np.trapezoid(np.abs(m - exact), x)) for x, m, exact in issue_runs]
    for coarse, fine in itertools.pairwise(w1):
        assert np.log2(coarse / fine) >= 0.8, w1
    for coarse, fine in itertools.pairwise(l1):
        assert np.log2(coarse / fine) >= 0.7, l1


def test_at_sigma_zero_the_stagnation_node_is_not_converged_pointwise(issue_runs):
    """RECORDED BEHAVIOUR, not a contract: the density at x = 1/2 stays above the exact peak by more than 1.1 at every
    level (measured at acdae412: 1.139, 1.171, 1.182; 1.186 at nx = 801), and it is the largest error. If this fails
    because the excess is gone, #2645's consistent deposit has landed: update `FPSLSolver`'s Convergence section,
    CONVENTIONS § 9, and this pin."""
    for x, m, exact in issue_runs:
        centre = int(np.argmin(np.abs(x - 0.5)))
        assert m[centre] - exact[centre] > 1.1, (m[centre], exact[centre])
        assert int(np.argmax(np.abs(m - exact))) == centre


def test_the_linear_deposit_gives_the_stagnation_node_twice_the_equations_inflow():
    """RECORDED BEHAVIOUR, not a contract (#2645). One step at σ = 0 on a flat density: the growth rate
    (M1 - M0) / (dt M0) is 2C at the node where the drift changes sign and C at every other interior node, where the
    continuity equation gives C everywhere; it does not depend on dt. If this fails with C at the stagnation node,
    #2645's consistent deposit has landed: update the documentation and this pin."""
    nx = 101
    x = np.linspace(0.0, 1.0, nx)
    for dt in (1e-2, 1e-3):
        M = _solve(nx, 1, dt, 0.0, np.ones(nx))
        rate = (M[1] - M[0]) / (dt * M[0])
        centre = int(np.argmin(np.abs(x - 0.5)))
        assert rate[centre] == pytest.approx(2 * C, rel=1e-10)
        interior = np.r_[2:centre, centre + 1 : nx - 2]
        np.testing.assert_allclose(rate[interior], C, rtol=1e-10)


def test_with_diffusion_the_density_converges_pointwise():
    """External oracle: σ = 0.1 on a Gaussian of variance v0 = 0.01 about 1/2, whose law under the drift -C (x - 1/2) is
    Gaussian about 1/2 with v(T) = v0 e^{-2CT} + (σ²/2C)(1 - e^{-2CT}) (Ornstein-Uhlenbeck); its tails at the walls are
    below 4e-6 of the peak. L∞ falls at order >= 0.8 along dt = h. Measured at acdae412: 1.02e-1, 5.39e-2, 2.77e-2
    (orders 0.92, 0.96)."""
    sigma, v0 = 0.1, 0.01
    errors = []
    for nx in (51, 101, 201):
        x = np.linspace(0.0, 1.0, nx)
        gauss = lambda v: np.exp(-((x - 0.5) ** 2) / (2 * v)) / np.sqrt(2 * np.pi * v)  # noqa: E731, B023
        m0 = gauss(v0) / np.trapezoid(gauss(v0), x)
        v_t = v0 * np.exp(-2 * C * T) + sigma**2 / (2 * C) * (1 - np.exp(-2 * C * T))
        exact = gauss(v_t) / np.trapezoid(gauss(v_t), x)
        errors.append(float(np.abs(_solve(nx, nx - 1, T, sigma, m0)[-1] - exact).max()))
    for coarse, fine in itertools.pairwise(errors):
        assert np.log2(coarse / fine) >= 0.8, errors
