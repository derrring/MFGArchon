"""`get_active_set` and `is_feasible` must agree about what "at the bound" means.

`is_feasible` tests one side of the inequality; `get_active_set` tested `|u - psi| < tol`, which is
two-sided. A point that VIOLATES the constraint was therefore reported INACTIVE — and an active-set
method runs on infeasible iterates by construction, since that is the state before projection. #1941
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary.constraints import BilateralConstraint, ObstacleConstraint


def test_a_violated_lower_obstacle_is_active():
    """`u[0] = -2` violates `lower = -1` by 1.0. The old two-sided form reported it inactive while
    `u[1] = -1`, exactly on the bound, was active — so the one point an active-set method must act
    on was the one it was told to ignore."""
    constraint = ObstacleConstraint(np.full(5, -1.0), constraint_type="lower")
    u = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    assert not constraint.is_feasible(u), "the premise of this test is that u is infeasible"
    np.testing.assert_array_equal(
        constraint.get_active_set(u),
        np.array([True, True, False, False, False]),
    )


def test_a_violated_upper_obstacle_is_active():
    """The mirror case. A fix applied to only one branch would pass the test above and fail here."""
    constraint = ObstacleConstraint(np.full(5, 1.0), constraint_type="upper")
    u = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])

    assert not constraint.is_feasible(u)
    np.testing.assert_array_equal(
        constraint.get_active_set(u),
        np.array([False, False, True, True, True]),
    )


def test_a_strictly_interior_field_has_an_empty_active_set():
    """Control. Without it, "everything is active" would satisfy both tests above."""
    constraint = ObstacleConstraint(np.full(5, -1.0), constraint_type="lower")
    u = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    assert constraint.is_feasible(u)
    assert not constraint.get_active_set(u).any()


@pytest.mark.parametrize("constraint_type", ["lower", "upper"])
def test_a_violation_smaller_than_the_tolerance_is_consistently_classified(constraint_type):
    """At `tol = 1e-10` a violation of `1e-9` used to be simultaneously infeasible and inactive.

    Whichever way the tolerance falls, the two predicates must agree: a point counted as violating
    must be counted as binding. This asserts the relationship, not a particular verdict, so it holds
    however the tolerance is later tuned.
    """
    obstacle = np.zeros(3)
    constraint = ObstacleConstraint(obstacle, constraint_type=constraint_type)
    epsilon = 1e-9
    u = np.full(3, -epsilon if constraint_type == "lower" else epsilon)

    tol = 1e-10
    if not constraint.is_feasible(u, tol=tol):
        assert constraint.get_active_set(u, tol=tol).all(), (
            "every point is infeasible, so every point must be reported as binding"
        )


def test_the_bilateral_constraint_inherits_the_same_agreement():
    """`BilateralConstraint` composes two obstacles; its active set must mark violations of either.

    `u[0]` is below the lower bound and `u[4]` above the upper, so a fix applied to one direction
    only is visible here.
    """
    constraint = BilateralConstraint(lower_bound=np.full(5, -1.0), upper_bound=np.full(5, 1.0))
    u = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    assert not constraint.is_feasible(u)
    active = constraint.get_active_set(u)
    assert active[0], "a point below the lower bound must be active"
    assert active[-1], "a point above the upper bound must be active"
    assert not active[2], "a strictly interior point must not be active"
