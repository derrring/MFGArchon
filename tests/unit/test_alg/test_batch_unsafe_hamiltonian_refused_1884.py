"""A Hamiltonian written for one point is refused on the 1-D batch path, not solved wrongly (#1884).

Since #1884 the 1-D HJB-FDM default evaluates the Hamiltonian on every node in one call. `0.5*p[0]**2` is a
correct Hamiltonian at one point and a wrong one in batch, where `p[0]` is node 0's momentum: with no
check, the solve converged cleanly to that wrong equation, 0.962 away from the per-point solve, with no
inner failure. `require_batch_safe_hamiltonian` compares three nodes evaluated together with each evaluated
alone, and refuses a difference.

One case per argument, because the probe has to vary each one: a point-only read of p, of x and of m.
Collapsing the probe's x to one node or holding its m constant left the x and m cases solving wrongly with
no error (0.202 and 0.057 off, review 2 of #2479), and a p-only pin passed both. A dead-zone cost read as
`p[0]` needs the probe's momenta to leave the dead zone: a probe with |p| <= 0.3 passed it, and it then
solved 1.894 off (review 3).

The check's own arithmetic is pinned apart, on a stand-in problem, since `MFGProblem` refuses a
Hamiltonian that is NaN or infinite at its validation points. NaN must match NaN, so a row-wise H that is
NaN outside its domain passes; and one infinite value must not make the tolerance infinite, so a point-only
H that is infinite at one probe node is still refused.

Oracle: the same Hamiltonian on the per-point path (`analytic_jacobian=False`) must reproduce the solve of
its row-wise twin on the default path, so the refusal is about batch evaluation and not about the
Hamiltonian. Deleting the check's call in `HJBFDMSolver.solve_hjb_system` makes the refusal not happen.
"""

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.base_hjb import require_batch_safe_hamiltonian
from mfgarchon.core.hamiltonian import HamiltonianBase
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

NX, NT = 41, 10
X = np.linspace(0.0, 1.0, NX)


def _row_wise(t, x, p, m):
    return 0.5 * np.sum(np.asarray(p) ** 2, axis=-1) + 0.3 * np.asarray(m) + 0.5 * np.asarray(x)[..., 0]


def _dead_zone_row_wise(t, x, p, m):
    return np.maximum(np.abs(np.asarray(p)[..., 0]) - 0.5, 0.0) + 0.3 * np.asarray(m)


# (point-only form, its row-wise twin). Each point-only form is correct at one point and reads node 0's value for
# every node in batch.
_CASES = {
    "p": (
        lambda t, x, p, m: 0.5 * np.asarray(p)[0] ** 2 + 0.3 * np.asarray(m) + 0.5 * np.asarray(x)[..., 0],
        _row_wise,
    ),
    "x": (
        lambda t, x, p, m: (
            0.5 * np.sum(np.asarray(p) ** 2, axis=-1) + 0.3 * np.asarray(m) + 0.5 * float(np.ravel(x)[0])
        ),
        _row_wise,
    ),
    "m": (
        lambda t, x, p, m: (
            0.5 * np.sum(np.asarray(p) ** 2, axis=-1) + 0.3 * float(np.ravel(m)[0]) + 0.5 * np.asarray(x)[..., 0]
        ),
        _row_wise,
    ),
    "p_dead_zone": (
        lambda t, x, p, m: np.maximum(np.abs(np.asarray(p)[0]) - 0.5, 0.0) + 0.3 * np.asarray(m),
        _dead_zone_row_wise,
    ),
}


class _H(HamiltonianBase):
    def __init__(self, formula, dp=None):
        super().__init__()
        self._formula = formula
        self._dp = dp

    def __call__(self, t, x, p, m):
        return self._formula(t, x, p, m)

    def dp(self, t, x, p, m):
        return super().dp(t, x, p, m) if self._dp is None else self._dp(t, x, p, m)


def _solve(hamiltonian, **solver_kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        try:
            problem = MFGProblem(
                model=Model(hamiltonian=hamiltonian, volatility=0.3),
                domain=TensorProductGrid(
                    bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1)
                ),
                conditions=Conditions(u_terminal=lambda x: np.cos(np.pi * x), m_initial=lambda x: 1.0, T=0.5),
                Nt=NT,
            )
            u_terminal = np.cos(np.pi * X)
            m = np.tile(1.0 + 0.5 * np.sin(np.pi * X), (NT + 1, 1))
            return HJBFDMSolver(problem, **solver_kwargs).solve_hjb_system(
                m, u_terminal, np.tile(u_terminal, (NT + 1, 1))
            )
        finally:
            logging.disable(logging.NOTSET)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_hamiltonian_written_for_one_point_is_refused_on_the_batch_path(case):
    point_only, row_wise = _CASES[case]
    reference = _solve(_H(row_wise))

    with pytest.raises(ValueError, match="differs from each node alone"):
        _solve(_H(point_only))

    np.testing.assert_allclose(_solve(_H(point_only), analytic_jacobian=False), reference, rtol=0, atol=1e-10)


def test_nan_matches_nan_and_an_infinite_value_does_not_make_the_check_vacuous():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1))

    def check(formula, dp=None):
        with np.errstate(invalid="ignore", divide="ignore"):
            require_batch_safe_hamiltonian(SimpleNamespace(hamiltonian_class=_H(formula, dp), geometry=grid))

    # Row-wise and NaN for |p| > 1, which two of the probe's three momenta are.
    check(lambda t, x, p, m: 1.0 - np.sqrt(1.0 - np.sum(np.asarray(p) ** 2, axis=-1)) + 0.3 * np.asarray(m))

    # Point-only in p and infinite at the probe node x = 0, with a row-wise dH/dp, so that only the comparison of H
    # itself can refuse it.
    with pytest.raises(ValueError, match="evaluate_H on 3 nodes together differs"):
        check(
            lambda t, x, p, m: 0.5 * np.asarray(p)[0] ** 2 + 1.0 / np.asarray(x)[..., 0],
            dp=lambda t, x, p, m: np.asarray(p, dtype=float),
        )
