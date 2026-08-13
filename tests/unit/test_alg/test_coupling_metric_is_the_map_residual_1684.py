"""The Picard metric measures the map's residual, not the damped step (#1684 item 7).

`FixedPointIterator` computed its convergence metrics from `self.U` -- the iterate AFTER damping
or Anderson -- against `U_old`. Under plain damping that is `theta * (U_new - U_old)`, so the
number the convergence criteria read carried the damping factor. Measured on the fixture below,
`l2distu_abs` at iteration 2, before the fix:

    relaxation   1.0        0.5        0.2        0.1
    reported     5.825e-01  3.357e-01  1.516e-01  7.944e-02

A factor of 7.3 bought by turning damping down. After the fix the same column runs the other way
-- 5.825e-01, 6.714e-01, 7.579e-01, 7.944e-01 -- because heavier damping has made LESS progress
and now says so.

The values do not become equal across relaxations, and a test asserting that would be wrong: each
damping follows a different path, so iteration 2 starts from a different `U_old`. What the fix
removes is the theta factor in the MEASUREMENT, and the observable consequence is the direction.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem():
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[21], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(
        geometry=grid,
        T=1.0,
        Nt=10,
        sigma=0.3,
        components=MFGComponents(
            m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(
                control_cost=QuadraticControlCost(control_cost=1.0),
                coupling=lambda m: m,
                coupling_dm=lambda m: 1.0,
            ),
        ),
    )


def _errors_at_iteration_two(relaxation: float, field: str = "U") -> float:
    problem = _problem()
    iterator = FixedPointIterator(
        problem, hjb_solver=HJBFDMSolver(problem), fp_solver=FPFDMSolver(problem), relaxation=relaxation
    )
    iterator.solve(max_iterations=3, tolerance=1e-14)
    history = iterator.l2distu_abs if field == "U" else iterator.l2distm_abs
    return float(history[1])


def test_heavier_damping_does_not_buy_a_smaller_reported_error():
    """The defect, as an ordering.

    Under it the reported error FELL monotonically as damping was turned down -- 5.825e-01 at
    theta 1.0 to 7.944e-02 at theta 0.1 -- so a user could reach any tolerance by damping harder.
    Fixed, the order inverts: less movement per iteration means a larger residual at the iterate
    reached, and the metric says so.
    """
    errors = {theta: _errors_at_iteration_two(theta) for theta in (1.0, 0.5, 0.2, 0.1)}

    assert errors[0.1] > errors[0.2] > errors[0.5], (
        "the reported error still falls with heavier damping, which is the defect: "
        + ", ".join(f"theta={t}: {e:.3e}" for t, e in errors.items())
    )


def test_the_reported_error_is_not_scaled_by_the_damping_factor():
    """The quantitative half: the theta factor is gone, not merely reduced.

    Under the defect `reported / theta` was roughly constant across dampings, because the
    measurement carried theta. It must not be: dividing the reported error by theta now spans
    more than an order of magnitude over this range (5.8e-01 to 7.9e+00), which a metric still
    proportional to theta cannot do.
    """
    scaled = {theta: _errors_at_iteration_two(theta) / theta for theta in (1.0, 0.1)}

    assert scaled[0.1] / scaled[1.0] > 5.0, (
        "reported/theta is still roughly constant across dampings, so the metric still carries "
        f"the damping factor: {scaled}"
    )


def test_the_first_iteration_metric_does_not_depend_on_the_damping_at_all():
    """The exact law, and the one that covers BOTH fields.

    At iteration 1 every damping starts from the same `U_old` and `M_old` -- the initial guesses --
    so the map output is bit-identical across dampings and the residual `||X_new - X_old||` must be
    too. Measured: 8.356340634e-01 and 6.806196935e-01 at theta 1.0, 0.5, 0.2 and 0.1, to every
    digit. Under the defect the metric carried theta and these spanned a factor of ten.

    This replaces an ordering assertion on the density that was simply wrong: the M residual at
    iteration 2 is NOT monotone in theta (0.7137, 0.7435, 0.7293, 0.7103), because by then the
    paths have diverged and M has its own damping factor. Found by mutation -- a half-fix leaving
    M damped passed the U-only tests, and the ordering test written to close that gap was red on
    the CORRECT tree.
    """
    values = {}
    for theta in (1.0, 0.5, 0.2, 0.1):
        problem = _problem()
        iterator = FixedPointIterator(
            problem, hjb_solver=HJBFDMSolver(problem), fp_solver=FPFDMSolver(problem), relaxation=theta
        )
        iterator.solve(max_iterations=1, tolerance=1e-14)
        values[theta] = (float(iterator.l2distu_abs[0]), float(iterator.l2distm_abs[0]))

    reference = values[1.0]
    for theta, (u_err, m_err) in values.items():
        assert u_err == pytest.approx(reference[0], rel=1e-12), f"theta={theta}: U metric moved with damping"
        assert m_err == pytest.approx(reference[1], rel=1e-12), f"theta={theta}: M metric moved with damping"
