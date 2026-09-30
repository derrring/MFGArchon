"""A keyword the weak-form solvers do not name raises, rather than being dropped (#2419).

``WeakFormFPSolver.solve_fp_system`` and ``WeakFormHJBSolver.solve_hjb_system`` ended in a
``**kwargs`` neither body read, and ``MeshlessGalerkinHJBSolver`` forwarded its own into the latter.
Measured at ``ac177dd4`` on the meshless pair: ``volatilty=3.0`` (misspelled) changed the solve by
exactly 0, where the spelled name changed it by 5e-1 (FP) and 5e-2 (HJB). ``FixedPointIterator``
warned on an unknown keyword and was built anyway.

Each test has its control beside it: the correctly spelled keyword must move the solve, so a probe
that could not see a volatility change cannot pass as a refusal.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.meshless_galerkin import MeshlessGalerkinFPSolver, MeshlessGalerkinHJBSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_N, _NT = 15, 10


def _setup(use_newton=False):
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_N], boundary_conditions=no_flux_bc(dimension=1))
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.3)
    conditions = Conditions(
        u_terminal=lambda x: 0.5 * (x - 0.5) ** 2, m_initial=lambda x: 1.0 + 0.5 * np.cos(np.pi * x), T=0.5
    )
    problem = MFGProblem(model=model, domain=grid, conditions=conditions, Nt=_NT)
    points = np.linspace(0.0, 1.0, _N).reshape(-1, 1)
    delta = 2.6 / np.sqrt(_N)
    return (
        problem,
        MeshlessGalerkinHJBSolver(problem, points, delta=delta, use_newton=use_newton),
        MeshlessGalerkinFPSolver(problem, points, delta=delta),
    )


def _arrays():
    x = np.linspace(0.0, 1.0, _N)
    return np.tile(1.0 + 0.5 * np.cos(np.pi * x), (_NT + 1, 1)), np.tile(0.5 * (x - 0.5) ** 2, (_NT + 1, 1))


def _solve(side, **keywords):
    _, hjb, fp = _setup()
    M, U = _arrays()
    if side == "hjb":
        return hjb.solve_hjb_system(M, U[-1], U, **keywords)
    return fp.solve_fp_system(M[0], potential_field=U, **keywords)


@pytest.mark.parametrize("side", ["hjb", "fp"])
def test_a_misspelled_keyword_raises_where_the_spelled_one_changes_the_solve(side):
    unchanged = _solve(side)
    # Control: the probe sees a volatility change when the name is right.
    assert np.max(np.abs(_solve(side, volatility=0.9) - unchanged)) > 1e-3
    with pytest.raises(TypeError, match="volatilty"):
        _solve(side, volatilty=0.9)


def test_the_fp_side_still_takes_the_base_interface_show_progress():
    # MultiPopulationIterator and the Newton path pass show_progress to every FP solver.
    np.testing.assert_array_equal(_solve("fp", show_progress=False), _solve("fp"))


def test_the_fixed_point_iterator_refuses_an_unknown_keyword_and_explains_a_removed_one():
    problem, hjb, fp = _setup()
    with pytest.raises(TypeError, match=r"unexpected keyword argument.*relaxtion"):
        FixedPointIterator(problem, hjb, fp, relaxtion=0.5)
    # A removed name keeps its curated migration message.
    with pytest.raises(ValueError, match=r"damping_factor.*relaxation"):
        FixedPointIterator(problem, hjb, fp, damping_factor=0.5)


def test_the_meshless_constructor_use_newton_is_the_solve_default():
    # MeshlessGalerkinHJBSolver's override that did this forwarded **kwargs; the parent now reads the
    # solver's _use_newton_default when solve_hjb_system is not told (#2419).
    _, hjb, _ = _setup(use_newton=True)
    M, U = _arrays()
    default = hjb.solve_hjb_system(M, U[-1], U)
    np.testing.assert_array_equal(default, hjb.solve_hjb_system(M, U[-1], U, use_newton=True))
    # Control: the two inner solvers do give different bits, so the equality above can fail.
    assert not np.array_equal(default, hjb.solve_hjb_system(M, U[-1], U, use_newton=False))
