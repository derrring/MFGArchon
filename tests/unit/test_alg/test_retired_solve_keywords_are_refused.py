"""Positional arguments the removal before 1.0 and #2977 deleted are Python's own `TypeError`.

`FixedPointIterator.solve` and `FictitiousPlayIterator.solve` take three positional arguments, so a fourth, where
`return_tuple` stood, is refused. Neither takes `**kwargs` since #2997, so a deleted keyword is Python's own
`TypeError` as well; that is not pinned here.

The two methods whose `coupling_coefficient` parameter #2977 deleted take every later parameter by keyword, so a
positional value meant for it is Python's own `TypeError` rather than the next parameter's value.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.numerical.flux_diagnostics import FluxDiagnostics

ITERATORS = [FixedPointIterator, FictitiousPlayIterator]


def _iterator(cls):
    problem = MFGProblem(
        model=Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.2),
        domain=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1)),
        conditions=Conditions(m_initial=lambda x: 1.0 + 0.0 * np.asarray(x[0]), u_terminal=lambda x: 0.0, T=0.2),
        Nt=4,
    )
    return cls(problem, HJBFDMSolver(problem), FPFDMSolver(problem))


@pytest.mark.parametrize("cls", ITERATORS, ids=[c.__name__ for c in ITERATORS])
def test_a_fourth_positional_argument_is_refused(cls):
    with pytest.raises(TypeError, match=r"takes .* positional argument"):
        _iterator(cls).solve(None, None, None, True)


def test_a_positional_value_cannot_slide_past_a_deleted_coupling_coefficient():
    """#2977: `build_advection_matrix(U, 2.0)` bound 2.0 to the deleted coefficient, and `compute_boundary_flux(m,
    None, u)` read u as the deleted field's; neither may now land in `time` or `diffusion`."""
    problem = _iterator(FixedPointIterator).problem
    grid = np.linspace(0.0, 1.0, 11)
    with pytest.raises(TypeError, match=r"takes 2 positional arguments but 3 were given"):
        HJBFDMSolver(problem).build_advection_matrix(grid**2, 2.0)
    with pytest.raises(TypeError, match=r"takes from 2 to 3 positional arguments but 4 were given"):
        FluxDiagnostics(dimension=1, spacing=(0.1,)).compute_boundary_flux(1.0 + grid, None, grid)
