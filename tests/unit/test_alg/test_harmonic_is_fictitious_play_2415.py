"""FictitiousPlayIterator runs fictitious play by default, and stops on the map's residual (#2415).

Fictitious play best-responds to the average of past densities, M <- M + (M_best - M)/(n+1):
Cardaliaguet-Hadikhanloo, *ESAIM: COCV* 23(2) (2017), (2.2) with Theorem 2.1 for the
second-order system (this fixture, volatility 0.1) and (3.3) with Theorem 3.1 for the first-order
one (#1914's sigma = 0 fixture). Those are the update rules; the theorems assume a regularizing
coupling on the torus, which neither fixture has (local coupling, walls).

The oracle is `_reference`: that update written out with the same HJB and FP solvers and no loop
machinery, with the step scaled by BASE (the iterator's `initial_learning_rate`) so that the old
0.01 floor bites within 14 sweeps: 0.1/(k+1) passes it at k = 10. At BASE = 1 it is (2.2)
exactly; any other BASE is the same average, reweighted.

Three defects, and the test that fails when each fix is reverted on its own (measured on the
fixture below, 14 sweeps, with that fix reverted to its `0f937601` form):

- The default `min_learning_rate=0.01` clamped the steps from k = 10 on, so the belief left the
  oracle by 1.88e-03 relative. Killed by `test_fictitious_play_follows_the_average`.
- Convergence was measured on the averaged step, which is exactly alpha_k times the map
  difference and so shrinks under a decaying alpha whether or not the belief is near a fixed
  point: the recorded `l2distm_rel` over the oracle's residual was 0.094, 0.047, 0.032 at
  k = 0, 1, 2 (alpha_k = 0.1, 0.05, 0.033; the relative norm divides by the new belief, hence not
  exactly alpha_k). On #1914's sigma = 0 fixture that let the default tolerance report
  converged=True with the map residual at 357x the tolerance. Killed by
  `test_fictitious_play_records_the_map_residual`.
- In hybrid mode (`damp_value_function=True`) U is averaged too, and its recorded change was the
  same alpha_k-scaled step: over the oracle's U residual it was 0.344 and 0.191 at k = 1, 2 (at
  k = 0 both read 1.0: the cold start is U = u_T = 0, and any relative change from zero is total).
  Killed by
  `test_hybrid_mode_records_the_u_map_residual`.

After the fixes all three agree with the oracle to round-off (1.2e-16, and residual ratios of
1.0 at every sweep), so the rtol of 1e-10 below sits seven orders under the smallest defect.
FixedPointIterator is not tested here: its harmonic schedule keeps its floor, by design (#2415).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling import FictitiousPlayIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.model import Conditions, Model
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.convergence import sweep_change

SWEEPS = 14
BASE = 0.1


def _problem() -> MFGProblem:
    # The volatility is Model's default (0.1), not written out, so the fixture reads the same
    # before and after #2378 phase 4 renamed the keyword.
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
    bump = lambda x: np.exp(-10 * (np.asarray(x, dtype=float) - 0.5) ** 2)  # noqa: E731
    mass = grid.integrate(bump(grid.get_spatial_grid()[:, 0]))
    return MFGProblem(
        model=Model(
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            )
        ),
        domain=grid,
        conditions=Conditions(m_initial=lambda x: bump(x) / mass, u_terminal=lambda x: 0.0, T=0.2),
        Nt=4,
    )


def _reference(m0: np.ndarray, u_terminal: np.ndarray, hybrid: bool) -> dict[str, np.ndarray]:
    """The averaged best response from the loop's cold start: the final belief and the map residuals."""
    problem = _problem()
    hjb, fp = HJBFDMSolver(problem), FPFDMSolver(problem)
    integrate, dt = problem.spatial_measure().integrate, problem.dt
    M = np.tile(m0, (problem.Nt + 1, 1))
    U = np.tile(u_terminal, (problem.Nt + 1, 1))
    residual_M, residual_U = [], []
    for k in range(SWEEPS):
        U_best = np.asarray(hjb.solve_hjb_system(M, u_terminal, U))
        M_best = np.asarray(fp.solve_fp_system(m0, potential_field=U_best))
        metrics = sweep_change(U_best, U, M_best, M, integrate, dt)
        residual_M.append(metrics["l2distm_rel"])
        residual_U.append(metrics["l2distu_rel"])
        weight = BASE / (k + 1)
        M = (1 - weight) * M + weight * M_best
        U = (1 - weight) * U + weight * U_best if hybrid else U_best
    return {"M": M, "residual_M": np.asarray(residual_M), "residual_U": np.asarray(residual_U)}


def _run(hybrid: bool):
    problem = _problem()
    iterator = FictitiousPlayIterator(
        problem,
        hjb_solver=HJBFDMSolver(problem),
        fp_solver=FPFDMSolver(problem),
        initial_learning_rate=BASE,
        damp_value_function=hybrid,
    )
    result = iterator.solve(max_iterations=SWEEPS)
    # The loop holds the initial row of M and the terminal row of U fixed, so the oracle takes its
    # boundary data from them rather than restating how a problem evaluates its components.
    reference = _reference(np.asarray(result.M)[0], np.asarray(result.U)[-1], hybrid)
    return iterator, result, reference


@pytest.fixture(scope="module")
def pure():
    return _run(hybrid=False)


@pytest.fixture(scope="module")
def hybrid():
    return _run(hybrid=True)


def test_fictitious_play_follows_the_average(pure):
    iterator, result, reference = pure
    assert result.iterations == SWEEPS
    np.testing.assert_allclose(iterator.learning_rate_history, BASE / (np.arange(SWEEPS) + 1), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.M), reference["M"], rtol=1e-10, atol=1e-12)


def test_fictitious_play_records_the_map_residual(pure):
    _, result, reference = pure
    assert result.iterations == SWEEPS
    np.testing.assert_allclose(np.asarray(result.metadata["l2distm_rel"]), reference["residual_M"], rtol=1e-10)


def test_hybrid_mode_records_the_u_map_residual(hybrid):
    _, result, reference = hybrid
    assert result.iterations == SWEEPS
    np.testing.assert_allclose(np.asarray(result.metadata["l2distu_rel"]), reference["residual_U"], rtol=1e-10)
