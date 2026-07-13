"""Issue #1553: HJBGFDMSolver(derivative_method='rbf') was a documented option that has been 100%
broken since #1526 -- the non-LCR weight path routes through NeighborhoodBuilder's Taylor-SVD builder,
which consumes SVD factors LocalRBFOperator.get_taylor_data does not provide (a dummy shim returning
None), so every real 'rbf' solve raised an undiagnostic ``'NoneType' object has no attribute 'T'`` deep
in Newton-Jacobian assembly. Until a genuine RBF weight-builder + obstacle_sdf threading + convergence
test land, 'rbf' must fail LOUD at construction (a half-working parallel path is worse than none).
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_problem import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem_and_points():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[13], boundary_conditions=no_flux_bc(dimension=1))
    ham = SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=1.0), coupling=lambda m: 0.0 * m)
    problem = MFGProblem(
        geometry=grid,
        T=0.1,
        Nt=4,
        diffusion=0.02,
        components=MFGComponents(hamiltonian=ham, u_terminal=lambda x: 0.0, m_initial=lambda x: 1.0),
    )
    points = np.linspace(0.0, 1.0, 13).reshape(-1, 1)
    return problem, points


def test_rbf_derivative_method_fails_loud_at_construction():
    """'rbf' must raise NotImplementedError at CONSTRUCTION (diagnostic, references #1553), not crash
    deep in _build_differentiation_matrices with an AttributeError. Discriminating: without the
    construction guard, HJBGFDMSolver(..., derivative_method='rbf') constructs fine and this raises
    nothing -- the crash only surfaces later inside a solve."""
    problem, points = _problem_and_points()
    with pytest.raises(NotImplementedError, match="1553"):
        HJBGFDMSolver(problem=problem, collocation_points=points, derivative_method="rbf")


def test_taylor_derivative_method_still_builds():
    """The supported path is unchanged: 'taylor' constructs and builds its differentiation matrices."""
    problem, points = _problem_and_points()
    solver = HJBGFDMSolver(problem=problem, collocation_points=points, derivative_method="taylor")
    solver._build_differentiation_matrices()  # must not raise


def test_unknown_derivative_method_fails_loud():
    """An unrecognized method fails loud at construction (via the operator-dispatch else-branch, which
    runs in __init__) -- a regression pin of that pre-existing contract, so #1553's rbf guard did not
    need to duplicate it."""
    problem, points = _problem_and_points()
    with pytest.raises(ValueError, match="Unknown derivative_method"):
        HJBGFDMSolver(problem=problem, collocation_points=points, derivative_method="bogus")
