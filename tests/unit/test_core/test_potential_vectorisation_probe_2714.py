"""The potential's vectorisation probe must not have d rows (#2714).

`SeparableHamiltonian` decides once, on its first batch call, whether the user's potential takes a
batch (N, d) -> (N,). It used to probe with two rows and accept a result of shape (2,). A potential
written per point indexes coordinates, x[0], x[1], which on a batch are its rows, so in 2-D it
returned row 0 squared plus row 1 squared -- shape (2,) -- and passed. Measured at `0b5481fd` on
`x[0]**2 + x[1]**2`: max error 0.071 at N = 2, and a broadcast ValueError at every N >= 3, on every
later batch call too, the decision being cached.

The oracle is the closed form of each potential, evaluated without the Hamiltonian.
"""

from __future__ import annotations

import math

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian


def _per_point(d):
    return lambda t, x: sum(x[i] ** 2 for i in range(d))


def _vectorised(t, x):
    return np.sum(np.asarray(x) ** 2, axis=-1)


def _potential_values(H, X):
    """V at the rows of X, read back through H = H_control(0) - V - f(0) with no control cost."""
    N = X.shape[0]
    return -np.asarray(H(t=0.0, x=X, p=np.zeros_like(X), m=np.zeros(N)), dtype=float)


@pytest.mark.parametrize("d", [1, 2, 3, 4])
@pytest.mark.parametrize("N", [1, 2, 3, 5])
def test_a_per_point_potential_is_evaluated_per_point(d, N):
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(), potential=_per_point(d))
    X = np.random.default_rng(N * 10 + d).random((N, d))

    np.testing.assert_allclose(_potential_values(H, X), (X**2).sum(axis=1), rtol=1e-12)
    assert H._potential_is_vectorized is False


@pytest.mark.parametrize("d", [1, 2, 3, 4])
@pytest.mark.parametrize("N", [1, 2, 3, 5])
def test_a_vectorised_potential_keeps_the_fast_path(d, N):
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(), potential=_vectorised)
    X = np.random.default_rng(N * 10 + d).random((N, d))

    np.testing.assert_allclose(_potential_values(H, X), (X**2).sum(axis=1), rtol=1e-12)
    assert H._potential_is_vectorized is True


def _exp_of_first(t, x):
    return math.exp(-x[0])


def _branch_on_first(t, x):
    return x[0] if x[0] > 0.5 else 1.0 - x[0]


@pytest.mark.parametrize(
    ("potential", "closed_form"),
    [
        (_exp_of_first, lambda X: np.exp(-X[:, 0])),
        (_branch_on_first, lambda X: np.where(X[:, 0] > 0.5, X[:, 0], 1.0 - X[:, 0])),
    ],
    ids=["TypeError", "ValueError"],
)
def test_a_potential_that_raises_on_a_batch_is_evaluated_per_point(potential, closed_form):
    """The probe's except branch. These pass at 0b5481fd too: they pin the branch, not #2714.

    Before the fix a 3-D per-point potential that indexes x[2] reached it through IndexError, the
    two-row probe having no row 2; with d + 1 rows nothing else in this file does.
    """
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(), potential=potential)
    X = np.random.default_rng(11).random((5, 2))

    np.testing.assert_allclose(_potential_values(H, X), closed_form(X), rtol=1e-12)
    assert H._potential_is_vectorized is False


def test_the_cached_decision_serves_a_later_batch_of_another_size():
    """The decision taken on the first batch is reused, so it must be right for every later N."""
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(), potential=_per_point(2))
    rng = np.random.default_rng(7)
    for N in (2, 7, 3):
        X = rng.random((N, 2))
        np.testing.assert_allclose(_potential_values(H, X), (X**2).sum(axis=1), rtol=1e-12)


def test_an_empty_batch_decides_nothing():
    H = SeparableHamiltonian(control_cost=QuadraticControlCost(), potential=_vectorised)

    assert H._evaluate_potential_batch(np.zeros((0, 2)), 0.0).shape == (0,)
    assert H._potential_is_vectorized is None
