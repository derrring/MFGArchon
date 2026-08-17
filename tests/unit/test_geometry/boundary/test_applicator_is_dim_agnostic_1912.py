"""One `apply`, any dimension, pinned against the output the four names produced. #1912

`FDMApplicator` carried `apply_1d`, `apply_2d`, `apply_3d` and `apply_nd`. Their bodies were the
same single line -- `pad_array_with_ghosts(field, boundary_conditions, ghost_depth=1, time=time)` --
and none read the dimension, `domain_bounds` or `config`. Measured before deletion: all four
returned byte-identical arrays on the same input, and so did the instance method `apply`.

The values below were captured by RUNNING that code, not derived from the surviving one. A
comparison between the old names and the new would be tautological now that only `apply` exists.

Why the split mattered. #1912 was a 2-D Robin that dropped `alpha` and `value` and returned the
Neumann mirror while the 1-D calculator computed the condition correctly. A boundary condition is a
statement about the normal derivative at a face -- the dimension chooses an axis and a side and
changes nothing else -- so a per-dimension ENTRY POINT is what made "correct in 1-D, wrong in 2-D"
a sentence one could say at all.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.boundary.applicator_fdm import FDMApplicator
from mfgarchon.geometry.boundary.calculators import RobinCalculator

_ALPHA, _BETA, _VALUE = 1.0, 1.0, 0.5

#: Captured at abd5e214, before the four names were deleted, from `FDMApplicator.apply_nd` on
#: `np.arange(25).reshape(5, 5)` with ROBIN(alpha=1, beta=1, value=0.5) on all four walls.
_CAPTURED_LEFT_GHOST = np.array([1.0 / 3.0, 2.0, 11.0 / 3.0, 16.0 / 3.0, 7.0])


def _robin_bc(dimension: int) -> BoundaryConditions:
    faces = [("x_min", "x_max"), ("y_min", "y_max"), ("z_min", "z_max")][:dimension]
    return BoundaryConditions(
        dimension=dimension,
        default_bc=BCType.ROBIN,
        segments=[
            BCSegment(name=f, bc_type=BCType.ROBIN, boundary=f, alpha=_ALPHA, beta=_BETA, value=_VALUE)
            for pair in faces
            for f in pair
        ],
    )


def test_apply_reproduces_the_pre_consolidation_capture():
    got = FDMApplicator(dimension=2).apply(np.arange(25, dtype=float).reshape(5, 5), _robin_bc(2))
    np.testing.assert_allclose(got[1:-1, 0], _CAPTURED_LEFT_GHOST, rtol=0, atol=1e-12)


def test_only_one_entry_point_survives():
    """Joining this class with a dimension in its name is the defect, not a style question."""
    named = sorted(n for n in dir(FDMApplicator) if n.startswith("apply"))
    assert named == ["apply"], f"a dimension-named entry point is back: {named}"


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_the_ghost_matches_the_one_dimensional_owner_in_every_dimension(dimension):
    """The external oracle: `RobinCalculator` owns the condition, and the applicator must agree
    with it in every dimension. This is what #1912 violated in 2-D while passing in 1-D."""
    shape = (5,) * dimension
    field = np.arange(int(np.prod(shape)), dtype=float).reshape(shape)
    padded = FDMApplicator(dimension=dimension).apply(field, _robin_bc(dimension))

    calc = RobinCalculator(alpha=_ALPHA, beta=_BETA, rhs_value=_VALUE)
    interior = field[(0,) + (slice(None),) * (dimension - 1)] if dimension > 1 else field[0]
    ghost = padded[(0,) + (slice(1, -1),) * (dimension - 1)] if dimension > 1 else padded[0]

    want = np.vectorize(lambda v: calc.compute(interior_value=v, dx=1.0, side="min"))(interior)
    np.testing.assert_allclose(np.asarray(ghost), np.asarray(want), rtol=0, atol=1e-12)
