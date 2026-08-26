"""`get_ghost_values_nd` must produce a Robin wall when asked for one. #1961

`_compute_ghost_pair` and `_compute_single_ghost` took `(bc_type, bc_value)` and nothing else, so
a Robin condition reached them with its coefficients already discarded. Both branches read

    elif bc_type == BCType.ROBIN:
        return u_next_left.copy(), u_prev_right.copy()

-- the adjacent interior cell, which is the *impermeable-wall mirror*. Measured on three
coefficient sets the ghost was `3.10000` every time, and the residual of the condition the caller
wrote ranged over 1.7 to 5.2.

`get_ghost_values_nd` is deprecated and kept until v0.25.0 (#1955). It is not in the package
`__all__` but is reachable as an attribute of `mfgarchon.geometry.boundary`, which is the surface
that notice applies to — so a user following the deprecation path, still calling it, got a wall
that was not the wall they asked for, with no signal beyond the deprecation warning.

**The oracle is the condition, not the other code path.** After this change the branch calls
`ghost_cell_robin`, so comparing it to `pad_array_with_ghosts` would be comparing an owner to
itself.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    dirichlet_bc,
    get_ghost_values_nd,
    neumann_bc,
    periodic_bc,
    robin_bc,
)

_U = np.array([2.5, 3.1, 4.0, 5.2, 6.6])
_DX = 0.25
_BOUNDS = np.array([[0.0, 1.0]])


def _ghosts(bc):
    """The function is deprecated by design; the warning is not what is under test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return get_ghost_values_nd(_U, bc, (_DX,))


def _scalar(ghost) -> float:
    return float(np.atleast_1d(ghost)[0])


def _residual(ghost: float, interior: float, alpha: float, beta: float, g: float) -> float:
    """`alpha*u_b + beta*du/dn - g`, outward normal, side-free on a cell-centred grid (#1907)."""
    return alpha * (ghost + interior) / 2 + beta * (ghost - interior) / _DX - g


@pytest.mark.parametrize(
    ("alpha", "beta", "g"),
    [
        (1.0, 0.3, 0.7),  # the case measured at ghost 3.10000, residual 2.82
        (1.0, 0.0, 0.7),  # beta = 0: Dirichlet in disguise, and it was still mirrored
        (0.0, 1.0, 0.7),  # pure flux
        (2.0, -0.4, 0.0),  # negative beta
        (0.5, 1.5, -0.3),  # negative g
    ],
)
@pytest.mark.parametrize("wall", ["min", "max"])
def test_a_uniform_robin_wall_satisfies_its_own_condition(alpha, beta, g, wall):
    """Both walls are parametrised because both were wrong, in different amounts -- 2.82 and 3.52
    on the first row. A single-wall test would have hidden half of it."""
    bc = robin_bc(dimension=1, alpha=alpha, beta=beta, value=g)
    bc.domain_bounds = _BOUNDS
    ghosts = _ghosts(bc)

    ghost, interior = (_scalar(ghosts[(0, 0)]), _U[0]) if wall == "min" else (_scalar(ghosts[(0, 1)]), _U[-1])

    assert _residual(ghost, interior, alpha, beta, g) == pytest.approx(0.0, abs=1e-12)


def test_the_coefficients_actually_reach_the_formula():
    """Negative control. Before the fix every coefficient pair produced the same ghost, so a test
    that only checked one pair would have passed over a function that ignored them entirely."""
    ghosts = []
    for alpha, beta in ((1.0, 0.3), (0.0, 1.0), (2.0, -0.4)):
        bc = robin_bc(dimension=1, alpha=alpha, beta=beta, value=0.7)
        bc.domain_bounds = _BOUNDS
        ghosts.append(_scalar(_ghosts(bc)[(0, 0)]))

    assert len(set(np.round(ghosts, 9))) == len(ghosts), "three coefficient pairs, three ghosts"


def test_a_robin_face_of_a_mixed_bc_reads_its_own_segment():
    """The mixed path goes through `_compute_single_ghost`, a different branch with the same
    defect. The Dirichlet face is asserted alongside so a fix that broke the per-face routing
    while repairing Robin would fail here."""
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="rob", bc_type=BCType.ROBIN, value=0.7, alpha=1.0, beta=0.3, boundary="x_min"),
            BCSegment(name="dir", bc_type=BCType.DIRICHLET, value=9.0, boundary="x_max"),
        ],
        dimension=1,
        default_bc=BCType.NO_FLUX,
        domain_bounds=_BOUNDS,
    )
    ghosts = _ghosts(bc)

    assert _residual(_scalar(ghosts[(0, 0)]), _U[0], 1.0, 0.3, 0.7) == pytest.approx(0.0, abs=1e-12)
    assert _scalar(ghosts[(0, 1)]) == pytest.approx(2 * 9.0 - _U[-1]), "the Dirichlet face is untouched"


def test_a_robin_condition_without_coefficients_refuses():
    """`BoundaryConditions` carries `default_bc` and `default_value` and no default alpha/beta, so
    a face that falls through to a ROBIN default has no coefficients to be honoured with.
    Fabricating `1.0, 0.0` would be a Dirichlet wall -- the #1558 failure, and the same hole
    #1956 hit on the particle side.

    The construction has to be genuinely mixed and genuinely fall through: a single-segment BC is
    `is_uniform` and takes the other path entirely, which is how the first version of this test
    passed for the wrong reason. Here `x_min` and `x_max` are covered and the two `y` faces are
    not, so they resolve to the ROBIN default.
    """
    bc = BoundaryConditions(
        segments=[
            BCSegment(name="d", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min"),
            BCSegment(name="n", bc_type=BCType.NO_FLUX, value=0.0, boundary="x_max"),
        ],
        dimension=2,
        default_bc=BCType.ROBIN,
        default_value=0.5,
        domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
    assert not bc.is_uniform, "the premise is that the per-face path runs"
    assert bc.get_bc_type_at_boundary("y_min") is BCType.ROBIN, "and that a face falls through to it"

    def _call() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            get_ghost_values_nd(np.ones((5, 5)), bc, (_DX, _DX))

    with pytest.raises(ValueError, match="alpha/beta"):
        _call()


@pytest.mark.parametrize(
    ("name", "factory", "expected"),
    [
        ("dirichlet", lambda: dirichlet_bc(dimension=1, value=2.5), (2.5, -1.6)),
        # #2067: was (2.75, 5.55), the retired `u[1] - 2*dx*g` / `u[-2] + 2*dx*g` form. The
        # branch now delegates to `ghost_cell_neumann`, so both walls read `u_int + dx*g`.
        ("neumann", lambda: neumann_bc(dimension=1, value=0.7), (2.675, 6.775)),
        ("periodic", lambda: periodic_bc(dimension=1), (6.6, 2.5)),
    ],
)
def test_the_other_members_are_byte_identical(name, factory, expected):
    """#1955 pinned this function byte-identical across the module's deletion, over five cases.

    #2067 moved the `neumann` row: the branch stopped restating `u_next -/+ 2*dx*g` and now calls
    `ghost_cell_neumann`. The other two rows are what still prove the rest of the function is
    untouched. This row is not decoration either -- it is currently the only assertion separating
    the `dx` separation from the `2*dx` vertex mirror, which the linear-field oracle below cannot
    do because both reproduce a linear field exactly."""
    bc = factory()
    bc.domain_bounds = _BOUNDS
    ghosts = _ghosts(bc)

    assert _scalar(ghosts[(0, 0)]) == pytest.approx(expected[0])
    assert _scalar(ghosts[(0, 1)]) == pytest.approx(expected[1])


@pytest.mark.parametrize("slope", [3.0, -1.7, 0.0])
def test_a_neumann_wall_reproduces_a_linear_field_exactly(slope):
    """#2067: an EXTERNAL oracle, because agreement with `ghost_cell_neumann` is now tautological.

    The branch used to restate its own arithmetic in the expectation, which is what let it hold a
    retired convention through two reviews. A linear field is the oracle any first-order ghost rule
    must reproduce exactly, and it discriminates BOTH things that were wrong:

    - the convention. `du/dn` at the low wall of `u = slope*x` is `-slope`; the old branch read `g`
      as `du/dx` and applied it with opposite signs, so feeding `du/dn` inverted the low wall.
    - the separation. The old form measured across `2*dx` from the SECOND interior cell, so it
      disagreed with the owner even at `g = 0`. On a CELL-CENTRED grid at dx = 0.1: 0.300 on
      `u = 3x`, 0.500 on `u = sin(2*pi*x)`. Only a constant field agrees, which is why the
      `slope = 0` case below cannot carry this test on its own and the other two are not
      decoration. (0.588 is the same quantity on a VERTEX layout, where the nodes sit at j*dx;
      `u = 3x` gives 0.300 on either, which is how the mismatched pair went unnoticed.)
    """
    dx = 0.25
    x = np.arange(5) * dx
    u = slope * x
    for wall, index, dudn in ((0, 0, -slope), (1, -1, +slope)):
        bc = neumann_bc(dimension=1, value=dudn)
        bc.domain_bounds = _BOUNDS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ghosts = get_ghost_values_nd(u, bc, (dx,))
        exact = slope * (x[index] - dx if wall == 0 else x[index] + dx)
        assert _scalar(ghosts[(0, wall)]) == pytest.approx(exact, abs=1e-12), (
            f"wall {wall}: ghost must continue u = {slope}*x exactly"
        )


@pytest.mark.parametrize("slope", [3.0, -1.7])
def test_a_neumann_face_on_a_MIXED_boundary_reproduces_a_linear_field(slope):
    """#2067: the OTHER Neumann branch. Without this it had zero discrimination in 6610 tests.

    `_compute_ghost_pair` handles a uniform BoundaryConditions; a non-uniform one routes each face
    through `_compute_single_ghost`, whose Neumann branch this PR also changed. Measured before
    adding this: replacing that branch's whole body with `return 12345.0` turned **nothing** red at
    full suite scope. `test_a_robin_condition_without_coefficients_refuses` executes the line and
    asserts nothing about its value -- a covered line with no discrimination.

    The oracle is the same linear field the uniform test uses, for the same reason: agreement with
    `ghost_cell_neumann` is tautological once the branch calls it.
    """
    dx = 0.25
    x = np.arange(5) * dx
    u = slope * x
    bc = BoundaryConditions(
        segments=[
            # du/dn at the min wall of u = slope*x is -slope; the max wall is Dirichlet so the two
            # faces cannot be uniform and each one goes through `_compute_single_ghost`.
            BCSegment(name="lo", bc_type=BCType.NEUMANN, value=-slope, boundary="x_min"),
            BCSegment(name="hi", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max"),
        ],
        dimension=1,
    )
    bc.domain_bounds = _BOUNDS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ghosts = get_ghost_values_nd(u, bc, (dx,))

    assert _scalar(ghosts[(0, 0)]) == pytest.approx(slope * (x[0] - dx), abs=1e-12), (
        "the mixed-boundary Neumann face must continue u = slope*x exactly"
    )
