"""``u_terminal`` and ``m_initial`` take ruling 8's order, time first (#2431, #2375 ruling 8).

The IC/BC adapter used to find a two-argument callable's order by probing, and tried ``f(x, t)``
first. Both orders return a number, so ``lambda t, x: x**2`` was read with x as time: ``u_T`` came
back identically ``T**2``. The order is now read from the parameter names, as for every other user
callable (``bind_user_callable``): a callable its names bind with a time slot is read at ``T``
(``u_terminal``) or ``0`` (``m_initial``), and one whose names put time after space is refused.

External oracle: the expected values are closed forms of the grid coordinates and ``T``, computed
without the adapter. Each callable depends on ``t``, so the tests pin WHEN it is read as well as
which argument is ``x``.
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


def _u_terminal(problem):
    return np.asarray(problem.u_terminal).reshape(-1)


@pytest.mark.parametrize("T", [1.0, 2.0])
@pytest.mark.parametrize(
    "u_terminal",
    [
        lambda t, x: x**2 + t,  # bound by name
        lambda x, *, t=0.0: x**2 + t,  # keyword-only time: read at T, not at its default
    ],
    ids=["t_x", "keyword_only_t"],
)
def test_a_time_first_terminal_condition_is_read_at_x_and_at_T(u_terminal, T):
    x = _GRID_1D.coordinates[0]
    np.testing.assert_array_equal(_u_terminal(_problem(_GRID_1D, u_terminal, _one, T=T)), x**2 + T)


def test_a_time_first_initial_density_is_read_at_x_and_at_0():
    x = _GRID_1D.coordinates[0]
    # Mass 1 on [0, 1] at t = 0, so the construction does not warn. Not symmetric in (t, x): read
    # with x as time it would give 0.5 + x**2.
    problem = _problem(_GRID_1D, lambda x: 0.0 * x, lambda t, x: 0.5 + x + t**2, T=2.0)
    np.testing.assert_array_equal(np.asarray(problem.m_initial).reshape(-1), 0.5 + x)


def test_validation_reads_a_time_first_terminal_condition_at_T():
    # Finite at T = 2 and not at t = 0: validating it at 0 refuses a well-posed condition.
    x = _GRID_1D.coordinates[0]
    np.testing.assert_array_equal(_u_terminal(_problem(_GRID_1D, lambda t, x: x**2 / t, _one, T=2.0)), x**2 / 2.0)


@pytest.mark.parametrize(
    "u_terminal",
    [
        lambda t, x: float(np.sum(np.asarray(x) ** 2)) + t,  # bound by name
        lambda tau, x: float(np.sum(np.asarray(x) ** 2)) + tau,  # bound by position: x names the order
    ],
    ids=["t_x", "tau_x"],
)
def test_a_time_first_condition_in_2d_is_read_at_the_point_not_as_expanded_coordinates(u_terminal):
    X, Y = np.meshgrid(*_GRID_2D.coordinates, indexing="ij")
    problem = _problem(_GRID_2D, u_terminal, _one, T=2.0)
    np.testing.assert_allclose(_u_terminal(problem), (X**2 + Y**2 + 2.0).reshape(-1), rtol=0, atol=1e-15)


def test_expanded_coordinates_in_2d_are_still_read_as_coordinates():
    X, Y = np.meshgrid(*_GRID_2D.coordinates, indexing="ij")
    with pytest.warns(DeprecationWarning, match=r"expanded coordinate signature f\(x, y\)"):
        problem = _problem(_GRID_2D, lambda x, y: x**2 + y**2, _one)
    np.testing.assert_allclose(_u_terminal(problem), (X**2 + Y**2).reshape(-1), rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    ("grid", "old_order"),
    [
        (_GRID_1D, lambda x, t: x**2 + t),
        (_GRID_1D, lambda x, t=0.0: x**2 + t),  # a default does not make the old order readable
        (_GRID_2D, lambda x, t: float(np.sum(np.asarray(x) ** 2)) + t),
    ],
    ids=["1d", "1d_default", "2d"],
)
def test_the_old_order_is_refused_rather_than_read_with_x_as_time(grid, old_order):
    with pytest.raises(ValidationError, match=r"u_terminal .* takes \(x, t\), which is out of order.*\(t, x\)"):
        _problem(grid, old_order, _one, T=2.0)
