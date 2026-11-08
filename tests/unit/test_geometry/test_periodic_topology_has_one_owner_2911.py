"""#2911: which axes of a domain wrap, and with what period, has one owner, and every reader reports what it says.

The oracles of the approved plan (#2910, issuecomment-6724735013), stage 1:
- cross-reader identity: on every spelling of a periodic boundary condition, each reader reports the axes and periods
  the rule gives. The rule, written here independently of the code: an axis wraps if and only if both of its faces
  are PERIODIC, however spelt, and its period is the span of the bounds;
- translation equivariance: a periodic field shifted by one cell along a wrapped axis gives an output shifted by one
  cell.
A cell red at stage 1's head would land as ``xfail(strict=True)`` naming the stage that moves its reader; every reader
here reads the owner at stage 1's head.
"""

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import BCSegment, BoundaryConditions, no_flux_bc, periodic_bc
from mfgarchon.geometry.boundary.conditions import periodic_axes, periodic_on_every_face, wrapping_axes
from mfgarchon.geometry.boundary.types import BCType

BOUNDS = [(0.0, 1.0), (0.0, 2.0)]  # different spans per axis, so a period read from the wrong axis shows
N = 9


def _faces(kinds):
    """One segment per face, ``kinds`` giving (min, max) per axis."""
    segments = []
    for axis, name in enumerate("xy"):
        for side, kind in zip(("min", "max"), kinds[axis], strict=True):
            segments.append(BCSegment(name=f"{name}{side}", bc_type=kind, boundary=f"{name}_{side}"))
    return BoundaryConditions(segments=segments, dimension=2)


P, W = BCType.PERIODIC, BCType.NO_FLUX
#: spelling -> (the boundary condition, the axes the rule says wrap)
SPELLINGS = {
    "uniform": (lambda: periodic_bc(dimension=2), (0, 1)),
    "per face": (lambda: _faces([(P, P), (P, P)]), (0, 1)),
    "default_bc": (lambda: BoundaryConditions(segments=[], default_bc=P, dimension=2), (0, 1)),
    "cylinder, x wraps": (lambda: _faces([(P, P), (W, W)]), (0,)),
    "cylinder, y wraps": (lambda: _faces([(W, W), (P, P)]), (1,)),
    "no-flux": (lambda: no_flux_bc(dimension=2), ()),
}
PERIODS = {0: 1.0, 1: 2.0}


def _grid(bc):
    return TensorProductGrid(bounds=BOUNDS, Nx_points=[N, N], boundary_conditions=bc)


def _wrapped_by_wrap_coordinates(grid):
    point = np.array([[1.25 * PERIODS[0], 1.25 * PERIODS[1]]])
    return tuple(axis for axis in range(2) if not np.isclose(grid.wrap_coordinates(point)[0, axis], point[0, axis]))


def _wrapped_by_distance(grid):
    """An axis wraps for the distance when two points 0.9 L apart along it come out 0.1 L apart."""
    wrapped = []
    for axis in range(2):
        a, b = np.zeros(2), np.zeros(2)
        b[axis] = 0.9 * PERIODS[axis]
        if np.isclose(float(grid.compute_periodic_distance(a, b)), 0.1 * PERIODS[axis]):
            wrapped.append(axis)
    return tuple(wrapped)


def _wrapped_by_laplacian(grid):
    """An axis wraps for the Laplacian when row 0 reads the far end of that axis."""
    operator = grid.get_laplacian_operator()
    row = np.array([float((operator @ e)[0]) for e in np.eye(N * N)])
    far = {0: (N - 2) * N, 1: N - 2}  # node 0's neighbour across the wrap on an endpoint-inclusive grid
    return tuple(axis for axis in range(2) if abs(row[far[axis]]) > 0)


READERS = {
    "the owner, wrapping_axes": lambda g, bc: wrapping_axes(bc, 2, g),
    "TensorProductGrid.periodic_dimensions": lambda g, bc: g.periodic_dimensions,
    "TensorProductGrid.wrap_coordinates": lambda g, bc: _wrapped_by_wrap_coordinates(g),
    "TensorProductGrid.compute_periodic_distance": lambda g, bc: _wrapped_by_distance(g),
    "the grid's Laplacian": lambda g, bc: _wrapped_by_laplacian(g),
}


@pytest.mark.parametrize("reader", list(READERS))
@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_every_reader_wraps_the_axes_the_rule_gives(spelling, reader):
    make, expected = SPELLINGS[spelling]
    bc = make()
    assert tuple(READERS[reader](_grid(bc), bc)) == expected


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_every_reader_takes_the_period_from_the_bounds(spelling):
    make, expected = SPELLINGS[spelling]
    bc = make()
    grid = _grid(bc)
    rule = {axis: PERIODS[axis] for axis in expected}
    assert periodic_axes(bc, 2, grid) == rule
    assert grid.get_periods() == rule
    assert periodic_on_every_face(bc, 2, grid) is (len(expected) == 2)


#: A cell red at stage 1's head: the reader still decides on its own, and the stage named moves it to the owner. When it
#: turns green, its strict XPASS fails the run and prints this reason, which says what to do.
STAGE_2 = pytest.mark.xfail(
    strict=True,
    reason="remove this xfail: stage 2 of #2910/#2911 moved this reader to the topology owner, so it now refuses too",
)


#: Every reader above, and the ones that answer a yes/no question about the wrap.
ONE_FACE_READERS = {
    **READERS,
    "periodic_on_every_face": lambda g, bc: periodic_on_every_face(bc, 2, g),
    "TensorProductGrid.get_periods": lambda g, bc: g.get_periods(),
}


@pytest.mark.parametrize(
    "reader",
    [pytest.param(r, marks=STAGE_2) if r == "the grid's Laplacian" else r for r in ONE_FACE_READERS],
)
def test_an_axis_periodic_on_one_face_is_refused_by_every_reader(reader):
    bc = _faces([(P, W), (W, W)])
    with pytest.raises(ValueError, match=r"periodic on the min face of axis 0 and not on its max face"):
        ONE_FACE_READERS[reader](_grid(bc), bc)


@pytest.mark.parametrize("spelling", [s for s, (_, axes) in SPELLINGS.items() if 0 in axes])
def test_a_shift_along_a_wrapped_axis_commutes_with_the_laplacian(spelling):
    grid = _grid(SPELLINGS[spelling][0]())
    operator = grid.get_laplacian_operator()
    rng = np.random.default_rng(0)
    distinct = rng.standard_normal((N - 1, N))  # axis 0 wraps: N - 1 distinct nodes, the last repeats node 0

    def full(part):
        return np.concatenate([part, part[:1]], axis=0)

    shifted = np.roll(distinct, 1, axis=0)
    out = (operator @ full(distinct).ravel()).reshape(N, N)[: N - 1]
    out_shifted = (operator @ full(shifted).ravel()).reshape(N, N)[: N - 1]
    np.testing.assert_allclose(out_shifted, np.roll(out, 1, axis=0), atol=1e-10)
