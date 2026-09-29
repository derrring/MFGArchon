"""``u_terminal`` and ``m_initial`` take ruling 8's order, time first (#2431, #2375 ruling 8).

The IC/BC adapter used to find a two-argument callable's order by probing, and tried ``f(x, t)``
first. Both orders return a number, so ``lambda t, x: x**2`` was read with x as time: ``u_T`` came
back identically ``T**2``. The order is now read from the parameter names, as for every other user
callable (``bind_user_callable``), and the old order is refused.

External oracle: the values below are closed forms of the grid coordinates, computed without the
adapter. ``T = 2`` separates the fix from the defect, which returned 4 there; on ``[0, 1]`` at
``T = 1`` the defect's constant 1 has the same maximum error as its first reading, 0.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import neumann_bc, no_flux_bc
from mfgarchon.utils.validation import ValidationError

_GRID_1D = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=neumann_bc(dimension=1))
_GRID_2D = TensorProductGrid(
    bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[5, 4], boundary_conditions=no_flux_bc(dimension=2)
)


def _problem(grid, u_terminal, m_initial, T=1.0):
    model = Model(hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(lambda_=1.0)), volatility=0.2)
    conditions = Conditions(u_terminal=u_terminal, m_initial=m_initial, T=T)
    return MFGProblem(model=model, domain=grid, conditions=conditions, Nt=4)


def _one(x):
    return 1.0 + 0.0 * float(np.sum(x))


@pytest.mark.parametrize("T", [1.0, 2.0])
def test_a_time_first_terminal_condition_is_read_at_x(T):
    x = _GRID_1D.coordinates[0]
    problem = _problem(_GRID_1D, lambda t, x: x**2, _one, T=T)
    np.testing.assert_array_equal(np.asarray(problem.u_terminal).reshape(-1), x**2)


def test_a_time_first_initial_density_is_read_at_x():
    x = _GRID_1D.coordinates[0]
    # Mass 1 on [0, 1], so the solve does not warn; the defect read it as the constant 0.5.
    problem = _problem(_GRID_1D, lambda x: 0.0 * x, lambda t, x: 0.5 + x)
    np.testing.assert_array_equal(np.asarray(problem.m_initial).reshape(-1), 0.5 + x)


def test_a_time_first_condition_in_2d_is_read_at_the_point_not_as_expanded_coordinates():
    X, Y = np.meshgrid(*_GRID_2D.coordinates, indexing="ij")
    problem = _problem(_GRID_2D, lambda t, x: float(np.sum(np.asarray(x) ** 2)), _one)
    np.testing.assert_allclose(
        np.asarray(problem.u_terminal).reshape(-1), (X**2 + Y**2).reshape(-1), rtol=0, atol=1e-15
    )


def test_expanded_coordinates_in_2d_are_still_read_as_coordinates():
    X, Y = np.meshgrid(*_GRID_2D.coordinates, indexing="ij")
    with pytest.warns(DeprecationWarning, match=r"expanded coordinate signature f\(x, y\)"):
        problem = _problem(_GRID_2D, lambda x, y: x**2 + y**2, _one)
    np.testing.assert_allclose(
        np.asarray(problem.u_terminal).reshape(-1), (X**2 + Y**2).reshape(-1), rtol=0, atol=1e-15
    )


@pytest.mark.parametrize(
    ("grid", "old_order"),
    [
        (_GRID_1D, lambda x, t: x**2),
        (_GRID_2D, lambda x, t: float(np.sum(np.asarray(x) ** 2))),
    ],
)
def test_the_old_order_is_refused_rather_than_read_with_x_as_time(grid, old_order):
    with pytest.raises(ValidationError, match=r"u_terminal .* takes \(x, t\), which is out of order.*\(t, x\)"):
        _problem(grid, old_order, _one)
