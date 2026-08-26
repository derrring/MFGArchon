"""#2129: `ghost_cell_neumann` cited a linear field as evidence, and a linear field cannot show it.

The docstring's "verified exact on 12 combinations against `u = a*x`" is true, and true of the
`2*dx` mirror this function replaced as well. So it separates nothing. These tests pin both halves:
the negative one, so nobody restores the citation as if it settled something, and the positive one,
so the property it was supposed to establish has a measurement behind it.

The discriminating quantity is not the ghost VALUE but the derivative the consuming stencil takes
from it. That consumer is a centred difference -- `operators/stencils/finite_difference.py:95`
computes `(u[i+1] - u[i-1]) / (2h)`.
"""

from __future__ import annotations

import itertools

import pytest

import numpy as np

from mfgarchon.geometry.boundary.ghost_cells import ghost_cell_neumann

_SLOPES = (3.0, -1.7, 0.0)


def _mirror(u_next: float, dudx: float, dx: float, outward_sign: float) -> float:
    """The retired `2*dx` form, kept here only as the thing the evidence must separate."""
    return u_next + outward_sign * 2.0 * dx * dudx


@pytest.mark.parametrize(("slope", "dx", "outward_sign"), list(itertools.product(_SLOPES, (0.25, 0.1), (1.0, -1.0))))
def test_a_linear_field_cannot_separate_the_two_ghost_rules(slope, dx, outward_sign):
    """The NEGATIVE half, and the reason this file exists.

    On `u = slope*x` the two forms are byte-identical at both centrings and both walls. The
    docstring's 12-combination check is exactly this input, so it certifies the retired rule as
    firmly as the shipped one.
    """
    # wall at x = 0, outward normal `outward_sign`; interior one step in, second interior two steps
    x_int = -outward_sign * dx
    u_int, u_next = slope * x_int, slope * (2.0 * x_int)
    dudn = outward_sign * slope

    shipped = ghost_cell_neumann(u_int, dudn, dx)
    retired = _mirror(u_next, slope, dx, outward_sign)
    assert shipped == pytest.approx(retired, abs=1e-12), (
        "if these ever differ on a linear field the docstring's evidence becomes meaningful; "
        "as long as they do not, it certifies both rules equally"
    )


def test_the_derivative_a_centred_stencil_takes_is_exact_at_cell_centring():
    """The POSITIVE half. Nonlinear, because that is what the linear field could not do.

    Cell-centred puts ghost at -dx/2 and interior at +dx/2, so the centred difference across them
    is centred on the wall itself and the shipped form makes it exact -- 0.0 at every dx, not a
    rate.
    """
    f = lambda x: np.sin(x) + 0.3 * x**2  # noqa: E731
    fp = lambda x: np.cos(x) + 0.6 * x  # noqa: E731

    for dx in (0.1, 0.05, 0.025, 0.0125):
        u_int = f(dx / 2)
        ghost = ghost_cell_neumann(u_int, -fp(0.0), dx)
        assert (u_int - ghost) / dx == pytest.approx(fp(0.0), abs=1e-12), f"dx={dx}"


def test_the_same_stencil_is_only_first_order_at_vertex_centring():
    """The other half of the positive result, and the one the package has never exercised.

    Vertex-centred puts the wall ON the interior node, so the centred difference spans 2*dx around
    it and the shipped form's one-sided ghost leaves an O(dx) error. The retired mirror is exact
    here -- the two forms are order mirror images. This is asserted rather than left in prose so
    that "one formula for both centrings" stays a recorded trade rather than an assumed equality.

    Nothing in `alg/` or `solvers/` requests `VERTEX_CENTERED`, so this order is currently unreached
    in production. That is the reason one formula is kept, and it is a fact about this package
    rather than about the mathematics -- if it stops being true, this test is the measurement to
    start from.
    """
    f = lambda x: np.sin(x) + 0.3 * x**2  # noqa: E731
    fp = lambda x: np.cos(x) + 0.6 * x  # noqa: E731

    errors = []
    for dx in (0.1, 0.05, 0.025, 0.0125):
        ghost = ghost_cell_neumann(f(0.0), -fp(0.0), dx)
        errors.append(abs((f(dx) - ghost) / (2.0 * dx) - fp(0.0)))

    rates = [np.log2(a / b) for a, b in itertools.pairwise(errors)]
    assert 0.8 < min(rates) < 1.3, f"expected first order at a vertex wall, measured {rates}"

    # the mirror, on the same stencil, is exact -- which is what makes this a trade and not a defect
    for dx in (0.1, 0.05, 0.025, 0.0125):
        mirror = _mirror(f(dx), fp(0.0), dx, -1.0)
        assert (f(dx) - mirror) / (2.0 * dx) == pytest.approx(fp(0.0), abs=1e-12), f"dx={dx}"
