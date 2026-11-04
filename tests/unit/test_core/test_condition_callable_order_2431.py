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

import functools
import math
import re

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
    return 1.0 + 0.0 * sum(x)


def _u_terminal(problem):
    return np.asarray(problem.u_terminal).reshape(-1)


def _exact(t, x):
    return x[0] ** 2 + 10.0 * t


@pytest.mark.parametrize("T", [1.0, 2.0])
@pytest.mark.parametrize(
    "u_terminal",
    [
        lambda t, x: x[0] ** 2 + t,
        lambda x, *, t=0.0: x[0] ** 2 + t,  # keyword-only time: read at T, not at its default
        lambda x, t: x[0] ** 2 + t,  # read by name, so the order it is written in does not matter
    ],
    ids=["t_x", "keyword_only_t", "x_t"],
)
def test_a_condition_that_declares_time_is_read_at_x_and_at_T(u_terminal, T):
    x = _GRID_1D.coordinates[0]
    np.testing.assert_array_equal(_u_terminal(_problem(_GRID_1D, u_terminal, _one, T=T)), x**2 + T)


def test_a_time_first_initial_density_is_read_at_x_and_at_0():
    x = _GRID_1D.coordinates[0]
    problem = _problem(_GRID_1D, lambda x: 0.0 * x[0], lambda t, x: 0.5 + x[0] + t**2, T=2.0)
    np.testing.assert_array_equal(np.asarray(problem.m_initial).reshape(-1), 0.5 + x)


def test_validation_reads_a_time_first_terminal_condition_at_T():
    # Finite at T = 2 and not at t = 0: validating it at 0 refuses a well-posed condition.
    x = _GRID_1D.coordinates[0]
    np.testing.assert_array_equal(_u_terminal(_problem(_GRID_1D, lambda t, x: x[0] ** 2 / t, _one, T=2.0)), x**2 / 2.0)


def test_a_time_first_condition_in_2d_reads_each_coordinate():
    X, Y = np.meshgrid(*_GRID_2D.coordinates, indexing="ij")
    problem = _problem(_GRID_2D, lambda t, x: x[0] ** 2 + x[1] ** 2 + t, _one, T=2.0)
    np.testing.assert_allclose(_u_terminal(problem), (X**2 + Y**2 + 2.0).reshape(-1), rtol=0, atol=1e-15)


def test_a_wraps_wrapper_is_read_by_its_own_names():
    # Its __wrapped__ reports (t, x); it takes x alone, and that is what it is passed.
    @functools.wraps(_exact)
    def at_half(x):
        return _exact(0.5, x)

    x = _GRID_1D.coordinates[0]
    np.testing.assert_array_equal(_u_terminal(_problem(_GRID_1D, at_half, _one, T=2.0)), x**2 + 5.0)


class _NotAFunction:
    def __call__(self, x):
        return 0.0


@pytest.mark.parametrize(
    ("grid", "unreadable", "reason"),
    [
        (_GRID_2D, lambda x, y: x + y, r"y has no default and names no slot"),
        (_GRID_2D, lambda x, tau: x[0] + tau, r"tau has no default and names no slot"),
        (_GRID_2D, lambda *c: c[0], r"nothing reaches a starred parameter"),
        (_GRID_2D, lambda x, **kw: x[0], r"nothing reaches a starred parameter"),
        (_GRID_1D, functools.partial(_exact, t=0.5), r"freezes t by keyword"),
        (_GRID_1D, _NotAFunction(), r"not a plain def or lambda"),
        (_GRID_1D, lambda t: t**2, r"takes time and no space"),
        (_GRID_1D, lambda t, x: math.cos(x[0]) + t, r"raised TypeError"),
    ],
    ids=["expanded", "unnamed_time", "varargs", "varkwargs", "partial_fixing_t", "instance", "time_only", "point_only"],
)
def test_a_condition_that_cannot_be_read_by_its_names_is_refused(grid, unreadable, reason):
    with pytest.raises((ValidationError, TypeError, ValueError), match=reason):
        _problem(grid, unreadable, _one, T=2.0)
