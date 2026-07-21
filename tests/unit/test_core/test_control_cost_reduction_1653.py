"""Every control cost must reduce `evaluate` over the component axis (Issue #1653).

`QuadraticControlCost.evaluate` summed over `axis=-1`; `L1ControlCost` and `BoundedControlCost`
returned one value per component. Every class's own `lagrangian` already summed, so the fork was
inside `evaluate` alone -- the convention was settled, one method disagreed with it.

Consequence, measured before the fix:

    L1 / Bounded, d=1  ->  SeparableHamiltonian returns an (N, N) matrix   [silent]
    L1 / Bounded, d=2  ->  ValueError: operands could not be broadcast     [loud]

The d=1 case is the dangerous one: a square array is a plausible shape for a Hamiltonian
evaluated on a grid, so nothing downstream had reason to object.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.core.hamiltonian import (
    BoundedControlCost,
    L1ControlCost,
    QuadraticControlCost,
    SeparableHamiltonian,
)

COSTS = [
    ("quadratic", QuadraticControlCost(control_cost=1.0)),
    ("l1", L1ControlCost(lambda_=0.5)),
    ("bounded", BoundedControlCost(max_control=1.0)),
]


@pytest.mark.parametrize(("label", "cost"), COSTS, ids=[c[0] for c in COSTS])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_hamiltonian_is_one_value_per_point(label, cost, d):
    """The symptom the fork produced, across the dimensions it produced it in."""
    n_points = 3
    hamiltonian = SeparableHamiltonian(control_cost=cost)
    x = np.zeros((n_points, d))
    m = np.ones(n_points)
    p = np.full((n_points, d), 0.7)

    result = hamiltonian(x, m, p)

    assert np.shape(result) == (n_points,), (
        f"{label} at d={d} returned shape {np.shape(result)}; an (N, N) result here is the "
        "pre-#1653 per-component reduction leaking into the Hamiltonian"
    )


@pytest.mark.parametrize(("label", "cost"), COSTS, ids=[c[0] for c in COSTS])
def test_evaluate_agrees_with_the_class_own_lagrangian_convention(label, cost):
    """`evaluate` and `lagrangian` must reduce the same way.

    This is what makes the convention a fact about the class rather than a choice: both are
    already `axis=-1` on every cost's `lagrangian`, so `evaluate` had a settled target.
    """
    batch = np.array([[0.4, 0.9], [1.3, 0.2], [0.0, 0.0]])

    assert np.shape(cost.evaluate(batch)) == (3,)
    assert np.shape(cost.lagrangian(batch)) == (3,)


@pytest.mark.parametrize("label", ["l1", "bounded"])
def test_moreau_yosida_does_not_double_count_the_base(label):
    """The wrapper summed the base term to paper over the fork; that must go with the fork.

    Leaving it would double-reduce, collapsing the batch axis and adding the base summed over
    *all* points to each point.
    """
    base = dict(COSTS)[label]
    epsilon = 0.1
    smooth = base.regularize(epsilon)

    p_batch = np.array([[0.8, 0.8], [1.2, 0.3], [2.0, 0.1]])
    q_batch = smooth._prox_h(p_batch)
    expected = base.evaluate(q_batch) + np.sum((p_batch - q_batch) ** 2, axis=-1) / (2 * epsilon)

    values = smooth.evaluate(p_batch)

    assert values.shape == (3,)
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=1e-15)
    assert not np.allclose(values, values[0]), (
        "every point got the same value, which is what summing the base over the batch produces"
    )
