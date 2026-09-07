"""`get_active_set` and `is_feasible` must agree about what "at the bound" means.

`is_feasible` tests one side of the inequality; `get_active_set` tested `|u - psi| < tol`, which is
two-sided. A point that VIOLATES the constraint was therefore reported INACTIVE -- and an active-set
method runs on infeasible iterates by construction, since that is the state before projection. So
the old form returned an empty active set on exactly the points that needed attention. #1941

ADMISSION (#2257). This is a consistency assertion, not a defect pin, and it carries no retirement
condition because the invariant is permanent: the two predicates are two routes to one inequality
and must not disagree, whatever the tolerance is later tuned to. The earlier version of this file
asserted hard-coded expected masks -- `[True, True, False, False, False]` and three more -- which
pinned a particular tolerance VERDICT alongside the invariant, so tuning `tol` would have turned it
red without anything being wrong. What is asserted here instead is the RELATION.

Pointwise feasibility is built by slicing the field into single-point constraints and asking the
same public `is_feasible`, rather than by restating its inequality in the test. A restatement can
agree with a broken implementation; a second route through the real predicate cannot.

Mutation, measured for #2257: `get_active_set` reverted to the two-sided `np.abs(u - psi) < tol` in
`ObstacleConstraint` and to `|u - bound| < tol` in `BilateralConstraint`, both at once, since the
bilateral copy is reached only through its own branch. Unmutated 15 passed; mutated 13 failed --
8 of 8 obstacle parametrizations, 4 of 4 bilateral, and the regional-mask test.

The two that survive are `test_a_field_with_slack_far_larger_than_the_tolerance_has_an_empty_active_set`,
and that is what they are for: the two-sided form is also correct on a strictly interior field, so a
survivor there says the control is measuring the vice it was written for (an all-True active set)
rather than the defect. A control that dies with the mutation would not have been one.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary.constraints import BilateralConstraint, ObstacleConstraint

_TOLS = [1e-12, 1e-10, 1e-6, 1e-3]


def _offsets(tol: float) -> np.ndarray:
    """Signed distances from the bound, straddling `tol` at every order that matters.

    Includes exactly-on-the-bound (0.0), inside and outside the tolerance band, and violations far
    larger than any tolerance -- the last being the class the two-sided form misreported.
    """
    return np.array([-1.0, -10 * tol, -1.5 * tol, -0.5 * tol, 0.0, 0.5 * tol, 1.5 * tol, 10 * tol, 1.0])


def _pointwise_infeasible(constraint_type: str, psi: np.ndarray, u: np.ndarray, tol: float) -> np.ndarray:
    """Per-point feasibility, obtained from `is_feasible` itself rather than restated.

    `is_feasible` reduces over the whole field (`np.all`), so there is no pointwise form to call.
    Slicing to length-1 constraints recovers one without the test owning a copy of the inequality.
    """
    return np.array(
        [
            not ObstacleConstraint(psi[i : i + 1], constraint_type=constraint_type).is_feasible(u[i : i + 1], tol=tol)
            for i in range(u.size)
        ]
    )


@pytest.mark.parametrize("constraint_type", ["lower", "upper"])
@pytest.mark.parametrize("tol", _TOLS)
def test_a_point_the_feasibility_test_rejects_must_be_reported_binding(constraint_type, tol):
    """The invariant. Violating implies active, on every point, at every tolerance.

    Kills the two-sided `np.abs(u - psi) < tol`: 8 of the 8 parametrizations fail under it, because
    the `-1.0` and `+1.0` offsets violate by far more than any `tol` in `_TOLS` and the two-sided
    form calls them inactive. Unmutated: 8 passed.
    """
    sign = 1.0 if constraint_type == "lower" else -1.0
    psi = np.zeros(9)
    u = sign * _offsets(tol)

    constraint = ObstacleConstraint(psi, constraint_type=constraint_type)
    violating = _pointwise_infeasible(constraint_type, psi, u, tol)
    active = constraint.get_active_set(u, tol=tol)

    # Controls. Without both, the implication is satisfiable by an empty antecedent or by an
    # all-True consequent, and the assertion below would hold while measuring nothing.
    assert violating.any(), f"the sweep contains no violating point at tol={tol}; it tests nothing"
    assert not violating.all(), f"the sweep contains no feasible point at tol={tol}; it tests nothing"

    assert np.all(active[violating]), (
        f"{constraint_type} bound, tol={tol}: points {np.flatnonzero(violating & ~active).tolist()} "
        f"are infeasible by is_feasible and inactive by get_active_set. An active-set method would "
        f"skip exactly the points that need projecting."
    )


@pytest.mark.parametrize("constraint_type", ["lower", "upper"])
def test_a_field_with_slack_far_larger_than_the_tolerance_has_an_empty_active_set(constraint_type):
    """The other half of the vice. Without it, `active = all True` satisfies the invariant above.

    The margin is 1.0 against tol <= 1e-3, so this asserts the contract and not a tolerance verdict.
    """
    psi = np.zeros(5)
    u = (1.0 if constraint_type == "lower" else -1.0) * np.linspace(1.0, 5.0, 5)

    constraint = ObstacleConstraint(psi, constraint_type=constraint_type)
    assert constraint.is_feasible(u), "the premise is that this field is strictly interior"
    assert not constraint.get_active_set(u).any()


@pytest.mark.parametrize("tol", _TOLS)
def test_the_bilateral_constraint_holds_the_same_relation_on_both_bounds(tol):
    """`BilateralConstraint` carried its own third copy of the two-sided form.

    A fix applied to `ObstacleConstraint` alone leaves this red, which is how the third copy was
    found. Kills the same mutation applied to the bilateral branch: 4 of 4 parametrizations.
    """
    offsets = _offsets(tol)
    lower = np.full(offsets.size, -1.0)
    upper = np.full(offsets.size, 1.0)
    constraint = BilateralConstraint(lower_bound=lower, upper_bound=upper)

    # Below the lower bound by `offsets`, and above the upper bound by the same amounts.
    for bound, sign, name in ((lower, -1.0, "lower"), (upper, 1.0, "upper")):
        u = bound + sign * offsets
        violating = np.array(
            [
                not BilateralConstraint(lower_bound=lower[i : i + 1], upper_bound=upper[i : i + 1]).is_feasible(
                    u[i : i + 1], tol=tol
                )
                for i in range(u.size)
            ]
        )
        active = constraint.get_active_set(u, tol=tol)

        assert violating.any(), f"{name}: no violating point at tol={tol}"
        assert not violating.all(), f"{name}: no feasible point at tol={tol}"
        assert np.all(active[violating]), (
            f"{name} bound, tol={tol}: {np.flatnonzero(violating & ~active).tolist()} infeasible and inactive"
        )


def test_a_regional_mask_excludes_a_point_from_both_predicates_together():
    """The mask must not make the two predicates disagree either.

    `is_feasible` ors in `~region`; `get_active_set` ands in `region`. A point outside the region is
    therefore feasible AND inactive -- consistent. A point inside it keeps the relation above.
    """
    psi = np.zeros(4)
    region = np.array([True, True, False, False])
    u = np.array([-1.0, 1.0, -1.0, 1.0])

    constraint = ObstacleConstraint(psi, constraint_type="lower", region=region)
    active = constraint.get_active_set(u)

    assert not constraint.is_feasible(u), "u[0] violates inside the region"
    assert active[0], "a violation inside the region must be reported binding"
    assert not active[2], "a point outside the region is excluded from both predicates"
