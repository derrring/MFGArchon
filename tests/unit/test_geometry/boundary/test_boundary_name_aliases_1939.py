"""One resolver for boundary names, and no guessing when a name is unknown.

`parse_boundary_face` normalises aliases -- "left" and "x_min" are the same face. The FDM ghost path
went through it; `get_bc_type_at_boundary`, `get_bc_value_at_boundary` and `BCSegment.matches_point`
compared the raw strings with `==`. So one `BoundaryConditions` object answered differently depending
on which route asked, and an absorbing wall declared with the alias `BCSegment`'s own docstring lists
first silently became no-flux on the query path. #1939
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    pad_array_with_ghosts,
)
from mfgarchon.geometry.boundary.types import BoundaryFace, parse_boundary_face

_U = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
_DX = 0.25


def _exit_on(boundary: str) -> BoundaryConditions:
    """A Dirichlet exit on one face, no-flux everywhere else.

    `default_bc` is NO_FLUX and the exit is DIRICHLET **on purpose**. With the two equal, the query
    path's fall-through returns the right answer for the wrong reason and the two routes agree
    whether or not the alias resolved -- the census hit exactly that and had to rebuild the fixture.
    """
    return BoundaryConditions(
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary=boundary),
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0),
        ],
        dimension=1,
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array([[0.0, 1.0]]),
    )


def _bc_type_the_ghost_path_applied(bc: BoundaryConditions) -> str:
    """Read back which condition the ghost path imposed at the low x wall.

    Dirichlet(0) gives `2*0 - u[0] = -u[0]`; no-flux mirrors to `u[0]`. The two are distinguishable
    only because `u[0] != 0`, which is why the field starts at 1.0 rather than 0.
    """
    ghost = pad_array_with_ghosts(_U, bc, spacing=_DX)[0]
    if np.isclose(ghost, -_U[0]):
        return "DIRICHLET"
    if np.isclose(ghost, _U[0]):
        return "NO_FLUX"
    raise AssertionError(f"ghost {ghost} matches neither convention; the probe cannot classify it")


@pytest.mark.parametrize(("spelling", "expected"), [("x_min", "DIRICHLET"), ("left", "DIRICHLET")])
def test_an_alias_resolves_on_the_query_path_as_well_as_the_ghost_path(spelling, expected):
    """`left` is an alias for `x_min` in `_BOUNDARY_STRING_TO_FACE`, and `BCSegment`'s docstring
    names it first. Before this fix the ghost path honoured it and the query path did not."""
    bc = _exit_on(spelling)

    assert _bc_type_the_ghost_path_applied(bc) == expected
    assert bc.get_bc_type_at_boundary("x_min").name == expected


@pytest.mark.parametrize("spelling", ["bottom", "y_min"])
def test_a_face_on_another_axis_is_not_matched(spelling):
    """Control, in the other direction. `bottom` and `y_min` are axis 1, so at the x_min wall both
    routes must say NO_FLUX. Without this, a resolver that matched everything would pass the test
    above."""
    bc = _exit_on(spelling)

    assert _bc_type_the_ghost_path_applied(bc) == "NO_FLUX"
    assert bc.get_bc_type_at_boundary("x_min").name == "NO_FLUX"


def test_the_two_routes_agree_on_the_value_as_well_as_the_type():
    """`get_bc_value_at_boundary` carried the same raw `==`."""
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=7.0, boundary="left"),
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0),
        ],
        dimension=1,
        default_bc=BCType.NO_FLUX,
        default_value=99.0,
        domain_bounds=np.array([[0.0, 1.0]]),
    )

    assert bc.get_bc_value_at_boundary("x_min") == pytest.approx(7.0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("x_min", BoundaryFace(0, "min")),
        ("left", BoundaryFace(0, "min")),
        ("bottom", BoundaryFace(1, "min")),
        ("w_min", BoundaryFace(3, "min")),
        ("axis3_min", BoundaryFace(3, "min")),
        ("dim3_min", BoundaryFace(3, "min")),
        ("axis7_max", BoundaryFace(7, "max")),
        ("dim7_max", BoundaryFace(7, "max")),
    ],
)
def test_every_spelling_in_the_tree_resolves_to_the_same_face(name, expected):
    """Three emitters produce three spellings for the same face past `z`: `BoundaryEntity.to_string`
    gives `w_min` for axis 3 and `axis{N}_{side}` beyond it, while
    `applicator_particle._get_boundary_id` gives `dim{N}_{side}`. Only the first two parsed.

    `dim3_min` previously resolved to **axis 0**, so a condition declared on the fourth axis was
    applied to the first while the axis it was written for took the default. That the tree has three
    emitters at all is a separate problem; this asserts there is one resolver.
    """
    assert parse_boundary_face(name) == expected


@pytest.mark.parametrize("name", ["inlet_min", "outlet_max", "wall_min"])
def test_an_unrecognised_prefix_resolves_to_nothing_rather_than_axis_zero(name):
    """The old last branch mapped any string ending `_min`/`_max` to `BoundaryFace(0, side)`.

    That turned a naming mismatch into a *misapplied* boundary condition rather than an unmatched
    one: the caller's segment silently governed axis 0. Returning `None` lets the caller fall through
    to the default it declared -- a wrong answer it asked for, rather than one the resolver invented.
    """
    assert parse_boundary_face(name) is None


def test_an_unresolvable_name_still_matches_itself():
    """The fall-back to string equality is deliberate: a name neither route can resolve should still
    match the identical string, so this change does not silently drop a working custom identifier."""
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="inlet", bc_type=BCType.DIRICHLET, value=5.0, boundary="inlet_min"),
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0),
        ],
        dimension=1,
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array([[0.0, 1.0]]),
    )

    assert parse_boundary_face("inlet_min") is None, "the premise of this test is that it resolves to nothing"
    assert bc.get_bc_type_at_boundary("inlet_min") is BCType.DIRICHLET
    assert bc.get_bc_type_at_boundary("x_min") is BCType.NO_FLUX


@pytest.mark.parametrize("queried_as", ["x_min", "left"])
def test_matches_point_resolves_aliases_too(queried_as):
    """`BCSegment.matches_point` carried the same raw `!=`, and it is the third route to the same
    question -- `get_bc_at_point` goes through it.

    The segment is declared as `left` and queried as both spellings; a raw comparison honours only
    the identical one. Without this test, reverting `matches_point` alone passes the whole rest of
    this file -- measured.
    """
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=7.0, boundary="left"),
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0),
        ],
        dimension=1,
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array([[0.0, 1.0]]),
    )

    assert bc.get_bc_at_point(np.array([0.0]), boundary_id=queried_as).bc_type is BCType.DIRICHLET


def test_matches_point_still_rejects_a_different_face():
    """Control for the test above, so a resolver that matched everything would not pass it."""
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=7.0, boundary="left"),
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0),
        ],
        dimension=1,
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array([[0.0, 1.0]]),
    )

    assert bc.get_bc_at_point(np.array([1.0]), boundary_id="x_max").bc_type is BCType.NO_FLUX


def test_a_faceless_segment_covers_every_face_and_is_not_the_default():
    """The catch-all branch of `_segment_covers` -- `segment.boundary is None` -- must return True.

    This is the consolidation's own new code: two inline copies became one method, and the rule for
    that is a pin that outlives the fork. Mutating this branch to `return False` left **all 6215
    tests green**, measured, because every other fixture here gives the catch-all `wall` segment the
    same `NO_FLUX` that `default_bc` carries -- so the segment and the fall-through are
    indistinguishable and either path produces the right answer.

    Here the two differ on both axes: the wall is `NO_FLUX` with value 3.0, and the default is
    `PERIODIC` with value 99.0. Only the segment can produce this answer.
    """
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=7.0, boundary="x_max"),
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=3.0),
        ],
        dimension=1,
        default_bc=BCType.PERIODIC,
        default_value=99.0,
        domain_bounds=np.array([[0.0, 1.0]]),
    )

    assert not bc.is_uniform, "a uniform BC would short-circuit before the branch under test"
    assert bc.get_bc_type_at_boundary("x_min") is BCType.NO_FLUX
    assert bc.get_bc_value_at_boundary("x_min") == pytest.approx(3.0)


def test_matches_point_distinguishes_the_axis_and_not_only_the_side():
    """`test_matches_point_still_rejects_a_different_face` varies only the SIDE -- `left` against
    `x_max`, both axis 0 -- so it cannot separate "faces are equal" from "sides are equal".

    Measured: mutating `matches_point`'s `mine != theirs` to `mine.side != theirs.side` survives that
    test and all 1007 geometry tests; at full-suite scope exactly one pre-existing test in another
    directory catches it. `y_min` is axis 1 and side "min", so only a comparison that looks at the
    axis can reject it.
    """
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=7.0, boundary="left"),
            BCSegment(name="wall", bc_type=BCType.NO_FLUX, value=0.0),
        ],
        dimension=2,
        default_bc=BCType.NO_FLUX,
        domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )

    assert bc.get_bc_at_point(np.array([0.0, 0.0]), boundary_id="y_min").bc_type is BCType.NO_FLUX
