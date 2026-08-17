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
        ("neumann", lambda: neumann_bc(dimension=1, value=0.7), (2.75, 5.55)),
        ("periodic", lambda: periodic_bc(dimension=1), (6.6, 2.5)),
    ],
)
def test_the_other_members_are_byte_identical(name, factory, expected):
    """#1955 pinned this function byte-identical across the module's deletion, over five cases.
    This change touches only the ROBIN branch, and these are the members that must prove it."""
    bc = factory()
    bc.domain_bounds = _BOUNDS
    ghosts = _ghosts(bc)

    assert _scalar(ghosts[(0, 0)]) == pytest.approx(expected[0])
    assert _scalar(ghosts[(0, 1)]) == pytest.approx(expected[1])
