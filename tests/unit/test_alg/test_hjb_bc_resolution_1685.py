"""`_get_bc_info_1d` must return what the BoundaryConditions object declares (Issue #1685).

The defect this pins: two fallback branches read `bc.default_type` and `bc.get_boundary_type()`,
neither of which exists, and both `AttributeError`s were swallowed by bare `except: pass`. Control
fell through to a hardcoded `return BCType.NEUMANN, 0.0`, so every BC whose segments carry no
`face` -- everything the `uniform_bc()` family builds -- silently became zero-Neumann.

The whole suite could not tell the bug from the fix: at the time this file was added, the gate
returned byte-identical counts with and without the defect. These tests are the discrimination.
"""

from __future__ import annotations

import pytest

from mfgarchon.alg.numerical.hjb_solvers.base_hjb import _get_bc_info_1d
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    ("factory", "expected_type", "expected_value"),
    [
        (lambda: dirichlet_bc(value=7.0, dimension=1), BCType.DIRICHLET, 7.0),
        (lambda: neumann_bc(value=5.0, dimension=1), BCType.NEUMANN, 5.0),
        (lambda: no_flux_bc(dimension=1), BCType.NO_FLUX, 0.0),
        (lambda: periodic_bc(dimension=1), BCType.PERIODIC, 0.0),
    ],
)
def test_uniform_bc_resolves_to_what_it_declares(factory, expected_type, expected_value, side):
    """A faceless (uniform) BC must resolve to its own type and value, not to Neumann-0."""
    bc_type, value, _alpha, _beta = _get_bc_info_1d(factory(), side)

    assert bc_type == expected_type, (
        f"{side}: resolved {bc_type} for a BC declaring {expected_type}; "
        "the pre-#1685 fallback returned NEUMANN for every one of these"
    )
    assert value == pytest.approx(expected_value), f"{side}: resolved value {value}"


def test_type_and_value_come_from_the_same_place():
    """Reading the type from `bc.default_bc` pairs it with another segment's value.

    `default_bc` is `BCType | None` and is independent of the segments, so a faceless DIRICHLET
    segment beside `default_bc=NEUMANN` resolved to `(NEUMANN, 9.0)` -- applying ``du/dn = 9.0``
    where the caller declared ``u = 9.0``. Both must come through the accessor.
    """
    bc = BoundaryConditions(
        default_bc=BCType.NEUMANN,
        segments=[BCSegment(name="wall", bc_type=BCType.DIRICHLET, value=9.0)],
        dimension=1,
    )

    bc_type, value, _alpha, _beta = _get_bc_info_1d(bc, "left")

    assert (bc_type, value) == (bc.get_bc_type_at_boundary("x_min"), 9.0), (
        f"resolved ({bc_type}, {value}); the accessor says {bc.get_bc_type_at_boundary('x_min')} for the same object"
    )


def test_faced_segment_still_wins_over_the_default():
    """The Priority-1 path is unchanged: an explicitly faced segment takes precedence."""
    bc = BoundaryConditions(
        default_bc=BCType.NO_FLUX,
        segments=[
            BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=3.0, boundary="x_min"),
        ],
        dimension=1,
    )

    left_type, left_value, _a, _b = _get_bc_info_1d(bc, "left")
    assert (left_type, left_value) == (BCType.DIRICHLET, 3.0)
