"""A negative initial density is refused on the direct FP-FDM call, not clipped (#1683).

`MFGProblem` refuses a negative `m_initial`. A direct `FPFDMSolver.solve_fp_system(M_initial)` call
bypasses that check, and the time-stepping loop then zeroed every negative entry it found: an initial
density of -0.5 at one node came back as 0.0, with no error and no warning. The library does not clip
on its own (#2485), so the caller's data is refused instead.

The zero-entry case is the control: the refusal is about the sign, so a density that touches zero must
still solve. Restoring the clip makes the refusal case fail.
"""

from __future__ import annotations

import logging
import warnings

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

NX, NT = 21, 5


def _solve(m_initial):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        try:
            problem = MFGProblem(
                model=Model(
                    hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.2
                ),
                domain=TensorProductGrid(
                    bounds=[(0.0, 1.0)], Nx_points=[NX], boundary_conditions=no_flux_bc(dimension=1)
                ),
                conditions=Conditions(m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, T=0.5),
                Nt=NT,
            )
            return FPFDMSolver(problem).solve_fp_system(m_initial, potential_field=np.zeros((NT + 1, NX)))
        finally:
            logging.disable(logging.NOTSET)


@pytest.mark.parametrize("value", [-0.5, -1e-12], ids=["order_one", "round_off"])
def test_a_negative_initial_density_is_refused(value):
    m_initial = np.ones(NX)
    m_initial[3] = value
    with pytest.raises(ValueError, match="contains negative values"):
        _solve(m_initial)


def test_an_initial_density_that_touches_zero_still_solves():
    m_initial = np.ones(NX)
    m_initial[3] = 0.0
    M = _solve(m_initial)
    np.testing.assert_array_equal(M[0], m_initial)
