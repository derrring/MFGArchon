"""A Hamiltonian written for one point is refused when a problem takes it, not solved wrongly (#1884, #2875).

Solvers evaluate the Hamiltonian on every node in one call. `0.5*p[0]**2` is a correct Hamiltonian at one point and a
wrong one in batch, where `p[0]` is node 0's momentum: with no check, the solve converged cleanly to that wrong
equation, 0.962 away from the per-point solve, with no inner failure. `refuse_a_hamiltonian_whose_reads_disagree`
compares three nodes evaluated together with each evaluated alone, and refuses a difference.

One case per argument, because the probe has to vary each one: a point-only read of p, of x and of m.
Collapsing the probe's x to one node or holding its m constant left the x and m cases solving wrongly with
no error (0.202 and 0.057 off, review 2 of #2479), and a p-only pin passed both. A dead-zone cost read as
`p[0]` needs the probe's momenta to leave the dead zone: a probe with |p| <= 0.3 passed it, and it then
solved 1.894 off (review 3).

The check's own arithmetic is pinned apart, on a stand-in problem, since `MFGProblem` refuses a
Hamiltonian that is NaN or infinite at its validation points. NaN must match NaN, so a row-wise H that is
NaN outside its domain passes; and one infinite value must not make the tolerance infinite, so a point-only
H that is infinite at one probe node is still refused.

Oracle: the row-wise twin of each refused form solves, so the refusal is about batch evaluation and not about the
Hamiltonian. Deleting the check's call in `MFGProblem` makes the refusal not happen.
"""

from __future__ import annotations

import logging
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import HamiltonianBase, refuse_a_hamiltonian_whose_reads_disagree
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types.callable_protocols import spread_samples

NX, NT = 41, 10
X = np.linspace(0.0, 1.0, NX)


def _row_wise(t, x, p, m):
    return 0.5 * np.sum(np.asarray(p) ** 2, axis=-1) + 0.3 * np.asarray(m) + 0.5 * np.asarray(x[0])


def _dead_zone_row_wise(t, x, p, m):
    return np.maximum(np.abs(np.asarray(p)[..., 0]) - 0.5, 0.0) + 0.3 * np.asarray(m)


# (point-only form, its row-wise twin). Each point-only form is correct at one point and reads node 0's value for
# every node in batch.
_CASES = {
    "p": (
        lambda t, x, p, m: 0.5 * np.asarray(p)[0] ** 2 + 0.3 * np.asarray(m) + 0.5 * np.asarray(x[0]),
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
            0.5 * np.sum(np.asarray(p) ** 2, axis=-1) + 0.3 * float(np.ravel(m)[0]) + 0.5 * np.asarray(x[0])
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
                conditions=Conditions(u_terminal=lambda x: np.cos(np.pi * x[0]), m_initial=lambda x: 1.0, T=0.5),
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
    assert np.all(np.isfinite(_solve(_H(row_wise))))

    with pytest.raises(ValueError, match="evaluate_H, at 3 samples"):
        _solve(_H(point_only))


def test_nan_matches_nan_and_an_infinite_value_does_not_make_the_check_vacuous():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1))

    def check(formula, dp=None):
        with np.errstate(invalid="ignore", divide="ignore"):
            refuse_a_hamiltonian_whose_reads_disagree(_H(formula, dp), grid.get_spatial_grid())

    # Row-wise and NaN for |p| > 1, which two of the probe's three momenta are.
    check(lambda t, x, p, m: 1.0 - np.sqrt(1.0 - np.sum(np.asarray(p) ** 2, axis=-1)) + 0.3 * np.asarray(m))

    # Point-only in p and infinite at a probe node, with a row-wise dH/dp, so that only the comparison of H
    # itself can refuse it.
    with pytest.raises(ValueError, match="evaluate_H, at 3 samples, reads differently"):
        check(
            lambda t, x, p, m: 0.5 * np.asarray(p)[0] ** 2 + 1.0 / (np.asarray(x[0]) - X[spread_samples(NX, 3)[0]]),
            dp=lambda t, x, p, m: np.asarray(p, dtype=float),
        )
