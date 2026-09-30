"""The semi-Lagrangian pair structures a step alike on both sides (#1880).

When a characteristic crosses more than one cell per step, the HJB half cuts the step into
sub-steps. The FP half used to make one forward splat. The coupled Picard map then amplified an
antisymmetric perturbation ~3.3x per sweep on a reflection-symmetric problem, where upwind FD damps it
at 0.89 and SL refined in dt at 0.2. The mirror mismatch, with FP sub-stepping while HJB does not, is
unstable too. So the FP half sub-steps exactly when its HJB half does.

Here: two mutation-verified pins of Picard stability, a property two independent schemes agree on for
this fixture and not a law of the problem; an external oracle for the FP half alone, the stationary
Ornstein-Uhlenbeck density; two checks that check_solver_duality sees a hand-built pair that does not
sub-step alike (#2440); and two pins of the sub-step cap (#2438).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_sl_characteristics import cfl_substeps
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.factory.scheme_factory import create_paired_solvers
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types import NumericalScheme
from mfgarchon.utils import DualityStatus, check_solver_duality

_H = SeparableHamiltonian(
    control_cost=QuadraticControlCost(lambda_=1.0), coupling=lambda m: -m, coupling_dm=lambda m: -1.0
)


def _problem(volatility=0.0, eps=0.0):
    """Invariant under x -> 1 - x when eps = 0; eps seeds an antisymmetric perturbation of m0."""
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    mass = grid.integrate(np.exp(-10 * (grid.coordinates[0] - 0.5) ** 2))

    def m0(x):
        x = np.asarray(x)
        return np.exp(-10 * (x - 0.5) ** 2) * (1 + eps * (x - 0.5)) / mass

    conditions = Conditions(m_initial=m0, u_terminal=lambda x: 0.0, T=1.0)
    return MFGProblem(model=Model(hamiltonian=_H, volatility=volatility), domain=grid, conditions=conditions, Nt=10)


def _asymmetry(M):
    M = np.asarray(M)
    return float(np.linalg.norm(M - M[:, ::-1]) / np.linalg.norm(M))


def test_a_seeded_asymmetry_decays_under_the_sl_pair():
    """The FP half's CFL number has a median of ~13 per step here. With the FP half unsub-stepped the
    asymmetry grew 5.5x per sweep (3.3x on the capability matrix's unnormalised copy of this fixture)."""

    def asym_after(sweeps):
        result = _problem(eps=1e-6).solve(
            scheme=NumericalScheme.SL_LINEAR, max_iterations=sweeps, tolerance=1e-30, verbose=False
        )
        return _asymmetry(result.M)

    growth = (asym_after(8) / asym_after(4)) ** 0.25
    assert growth < 1.0, f"seeded asymmetry grows {growth:.3f}x per sweep"


def test_the_pair_stays_stable_when_its_hjb_half_does_not_substep():
    """The mirror mismatch: FP sub-stepping while HJB does not. This pair converges in 60 sweeps, before
    #1880 and after; forcing the FP half to sub-step regardless leaves it unconverged with asymmetry ~1
    after 200. The budget is not the claim, so it has room."""
    problem = _problem(volatility=0.2)
    hjb, fp = create_paired_solvers(
        problem, NumericalScheme.SL_LINEAR, hjb_config={"enable_adaptive_substepping": False}
    )
    result = problem.solve(hjb_solver=hjb, fp_solver=fp, max_iterations=100, tolerance=1e-8, verbose=False)
    assert result.converged
    assert _asymmetry(result.M) < 1e-10


@pytest.mark.parametrize(
    ("hjb_config", "fp_config"),
    [
        ({"enable_adaptive_substepping": False}, {}),
        ({}, {"enable_adaptive_substepping": False}),
    ],
)
def test_the_duality_check_flags_a_hand_built_pair_that_does_not_substep_alike(hjb_config, fp_config):
    """The first pair converges in 60 sweeps before #1880's fix and not after (asymmetry 0.95 after 200);
    the second is #1880 itself. check_solver_duality called both DISCRETE_DUAL, so Expert Mode was
    silent (#2440)."""
    problem = _problem(volatility=0.2)
    hjb, fp = HJBSemiLagrangianSolver(problem, **hjb_config), FPSLSolver(problem, **fp_config)
    with pytest.warns(UserWarning, match="does not sub-step alike"):
        assert check_solver_duality(hjb, fp).status == DualityStatus.NOT_DUAL
    # Control: the factory hands the FP half the HJB half's schedule.
    matched = create_paired_solvers(problem, NumericalScheme.SL_LINEAR, hjb_config=hjb_config)
    assert check_solver_duality(*matched, warn_on_mismatch=False).status == DualityStatus.DISCRETE_DUAL
    if fp_config:  # and it refuses a caller's fp_config that undoes that, without calling it a bug
        with (
            pytest.raises(ValueError, match="hjb_config and fp_config"),
            pytest.warns(UserWarning, match="does not sub-step alike"),
        ):
            create_paired_solvers(problem, NumericalScheme.SL_LINEAR, fp_config=fp_config)


def test_the_duality_check_ignores_how_many_substeps_a_half_takes():
    """An FP half at cfl_target=0.45 took 1609 sub-steps against 825 on #1880's fixture, and the seeded
    asymmetry still decayed at 0.978 per sweep, as in the matched pair (#2440)."""
    problem = _problem(volatility=0.2)
    pair = HJBSemiLagrangianSolver(problem), FPSLSolver(problem, cfl_target=0.45)
    assert check_solver_duality(*pair, warn_on_mismatch=False).status == DualityStatus.DISCRETE_DUAL
    # The factory copies the caller's config: writing the HJB schedule into it made the next pair built
    # from the same dict with a different HJB half a mismatch the caller never asked for.
    config = {"cfl_target": 0.45}
    create_paired_solvers(problem, NumericalScheme.SL_LINEAR, fp_config=config)
    assert config == {"cfl_target": 0.45}


def _ou_error(dimension, cfl):
    """FP half alone, U = a|x - c|^2: the stationary no-flux density is exp(-a|x - c|^2 / D)."""
    sigma, n = 0.4, 41
    if dimension == 1:
        grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[n], boundary_conditions=no_flux_bc(dimension=1))
        c, Nt, T = np.array([0.35]), 20, 1.0
    else:
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[n, n], boundary_conditions=no_flux_bc(dimension=2)
        )
        c, Nt, T = np.array([0.4, 0.55]), 12, 0.6
    axes = np.meshgrid(*grid.coordinates, indexing="ij")
    Q = sum((X - ci) ** 2 for X, ci in zip(axes, c, strict=True))
    d = grid.coordinates[0][1] - grid.coordinates[0][0]
    slope = max(float(np.max(np.abs(g))) for g in np.atleast_1d(np.gradient(Q, *([d] * dimension))))
    a = cfl / (slope * (T / Nt) / d)  # the largest grid velocity crosses `cfl` cells per step
    problem = MFGProblem(
        model=Model(hamiltonian=_H, volatility=sigma),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=T),
        Nt=Nt,
    )
    m0 = np.exp(-30 * sum((X - 0.5) ** 2 for X in axes))
    M = FPSLSolver(problem).solve_fp_system(m0, potential_field=np.broadcast_to(a * Q, (Nt + 1, *Q.shape)))
    stationary = np.exp(-a * Q / (sigma**2 / 2))
    stationary *= grid.integrate(M[-1]) / grid.integrate(stationary)
    return grid.integrate(np.abs(M[-1] - stationary)) / grid.integrate(stationary)


@pytest.mark.parametrize(("dimension", "cfl", "bound"), [(1, 10.0, 0.05), (2, 8.0, 0.08)])
def test_the_fp_half_relaxes_to_the_stationary_ou_density_at_cfl_above_one(dimension, cfl, bound):
    # Relative L1, measured 1-D / 2-D: 0.036 / 0.060, half what 21 points and half the steps give, so a
    # first-order discretisation error. Outside the bound: one splat per step 0.150 / 0.192, drift x1.25
    # 0.078 / 0.111, diffusion x0.75 0.108 / 0.154. Not pinned here: the sub-step count. Half as many,
    # or at most 3, stay inside the bound.
    assert _ou_error(dimension, cfl) < bound


@pytest.mark.parametrize("half", ["hjb", "fp"])
def test_a_step_needing_more_substeps_than_the_cap_is_refused(half):
    """A capped HJB step ran away on the #1880 fixture at 41 points: from a sub-step CFL of 2 on, U grew
    ever faster until the solve ended in NaN after a warning. Both halves now refuse the
    step (#2438)."""
    problem = _problem()
    x = np.linspace(0.0, 1.0, 21)
    U = np.tile(5.0 * (x - 0.5) ** 2, (11, 1))  # max|U_x| ~5: CFL ~10 at dt = 0.1, dx = 0.05
    M = np.ones_like(U)

    def solve(cap):
        if half == "hjb":
            return HJBSemiLagrangianSolver(problem, max_substeps=cap).solve_hjb_system(M, U[-1], U)
        return FPSLSolver(problem, max_substeps=cap).solve_fp_system(M[0], potential_field=U)

    with pytest.raises(ValueError, match=r"needs \d+ sub-steps .*max_substeps=3\b"):
        solve(3)
    # Control: with room for the schedule, the same solve completes.
    assert np.isfinite(solve(100)).all()


def test_the_cap_admits_a_step_that_needs_exactly_max_substeps():
    """The refusal is at needs > max_substeps: not at >=, and not on the capped sub-step's CFL number,
    which here would be 0.56 (#2438)."""
    assert cfl_substeps(5.0, cfl_target=0.5, max_substeps=10) == 10
    with pytest.raises(ValueError, match=r"needs 10 sub-steps .*max_substeps=9\b"):
        cfl_substeps(5.0, cfl_target=0.5, max_substeps=9)
    # A sub-step never exceeds cfl_target: 10.2 needed rounds up to 11, not down to 10.
    assert 5.1 / cfl_substeps(5.1, cfl_target=0.5, max_substeps=11) <= 0.5
