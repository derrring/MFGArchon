"""A Hamiltonian written for one point is refused on the 1-D batch path, not solved wrongly (#1884).

Since #1884 the 1-D HJB-FDM default evaluates the Hamiltonian on every node in one call. `0.5*p[0]**2` is a
correct Hamiltonian at one point and a wrong one in batch, where `p[0]` is node 0's momentum: with no
check, the solve converged cleanly to that wrong equation, 0.962 away from the per-point solve, with no
inner failure. `require_batch_safe_hamiltonian` compares three nodes evaluated together with each evaluated
alone, and refuses a difference.

Oracle: the same Hamiltonian on the per-point path (`analytic_jacobian=False`) must reproduce the solve of
its row-wise twin on the default path, so the refusal is about batch evaluation and not about the
Hamiltonian. Deleting the check's call in `HJBFDMSolver.solve_hjb_system` makes the refusal not happen.
"""

from __future__ import annotations

import logging
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import HamiltonianBase
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

NX, NT = 41, 10
X = np.linspace(0.0, 1.0, NX)


class _PointOnly(HamiltonianBase):
    """Correct at one point; in batch, p[0] is node 0's momentum row."""

    def __call__(self, t, x, p, m):
        return 0.5 * np.asarray(p)[0] ** 2 + 0.3 * np.asarray(m)


class _RowWise(HamiltonianBase):
    """The same Hamiltonian, written row-wise."""

    def __call__(self, t, x, p, m):
        return 0.5 * np.sum(np.asarray(p) ** 2, axis=-1) + 0.3 * np.asarray(m)


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


def test_a_hamiltonian_written_for_one_point_is_refused_on_the_batch_path():
    reference = _solve(_RowWise())

    with pytest.raises(ValueError, match="differs from each node alone"):
        _solve(_PointOnly())

    np.testing.assert_allclose(_solve(_PointOnly(), analytic_jacobian=False), reference, rtol=0, atol=1e-10)
