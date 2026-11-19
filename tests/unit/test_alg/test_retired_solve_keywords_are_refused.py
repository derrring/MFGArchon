"""The two iterators' `solve()` refuse the keywords the removal before 1.0 deleted, by name.

`FixedPointIterator.solve` and `FictitiousPlayIterator.solve` keep a `**kwargs` (`problem.solve()` passes `verbose`
through it), so a deleted keyword would otherwise be accepted in silence: `return_tuple=True` returned a
`SolverResult` and `Niter_max=2` was ignored. Each of the five is refused by name, and a fourth positional argument,
where `return_tuple` stood, is Python's own `TypeError`.

The two methods whose `coupling_coefficient` parameter #2977 deleted take every later parameter by keyword, so a
positional value meant for it is Python's own `TypeError` rather than the next parameter's value.
"""

from __future__ import annotations

import re

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

RETIRED = [
    ("return_tuple", True, "solve() returns a SolverResult"),
    ("Niter_max", 2, "pass max_iterations="),
    ("max_picard_iterations", 2, "pass max_iterations="),
    ("l2errBoundPicard", 1e-3, "pass tolerance="),
    ("picard_tolerance", 1e-3, "pass tolerance="),
]


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


@pytest.mark.parametrize(("name", "value", "replacement"), RETIRED, ids=[r[0] for r in RETIRED])
@pytest.mark.parametrize("cls", ITERATORS, ids=[c.__name__ for c in ITERATORS])
def test_each_retired_keyword_is_refused_by_name(cls, name, value, replacement):
    with pytest.raises(
        TypeError, match=rf"{cls.__name__}\.solve\(\) no longer takes {name}=: {re.escape(replacement)}"
    ):
        _iterator(cls).solve(**{name: value})


def test_a_positional_value_cannot_slide_past_a_deleted_coupling_coefficient():
    """#2977: `build_advection_matrix(U, 2.0)` bound 2.0 to the deleted coefficient, and `compute_boundary_flux(m,
    None, u)` read u as the deleted field's; neither may now land in `time` or `diffusion`."""
    problem = _iterator(FixedPointIterator).problem
    grid = np.linspace(0.0, 1.0, 11)
    with pytest.raises(TypeError, match=r"takes 2 positional arguments but 3 were given"):
        HJBFDMSolver(problem).build_advection_matrix(grid**2, 2.0)
    with pytest.raises(TypeError, match=r"takes from 2 to 3 positional arguments but 4 were given"):
        FluxDiagnostics(dimension=1, spacing=(0.1,)).compute_boundary_flux(1.0 + grid, None, grid)
