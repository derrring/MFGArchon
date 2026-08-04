"""A periodic ghost cell must hold the analytic periodic continuation, at every ghost depth.

`TensorProductGrid` is endpoint-inclusive, so `x[0]` and `x[-1]` are the same physical point and
the node one step left of `x[0]` is `x[-2]`. Both ghost paths in `applicator_fdm.py` -- the uniform
one and the mixed one, each with its own copy of the arithmetic -- read the last `g` interior
entries instead, which is the halo convention (N distinct DOFs over a period of `L + dx`). That
puts the duplicated node in the ghost and shifts the stencil by one cell.

Measured before the fix, against `sin(2 pi x)` on `Nx=21`: the ghost was wrong by **3.09e-01 at
g = 1, 2 and 3**, and `HJBWENOSolver` turned exactly periodic data (seam 2.4e-16) into a field with
a seam of 2.63e-01. After: ghost error ~5e-16 and the solver's seam is exactly 0.

The oracle is EXTERNAL -- `sin(2 pi x)` evaluated off-grid at the ghost coordinates -- not a second
implementation of the fill, so it cannot go tautological if the fill is later rewritten. It is also
what a same-side pin would have missed: `slice(-2g, -g)` and the correct `slice(-2g-1, -g-1)` agree
on nothing, but "ghost equals some interior value" is true of both.

Issue #1822; same convention #1829 settled for `enforce_periodic_value_nd`.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
from mfgarchon.geometry.boundary.applicator_fdm import PreallocatedGhostBuffer
from mfgarchon.geometry.boundary.types import PeriodicGridConvention

NX = 21
BOUNDS = (0.0, 1.0)


def _periodic_bc(uniform: bool) -> BoundaryConditions:
    """One segment takes `update_ghosts`' uniform path; two take the mixed one.

    Both are exercised because both carried the defect. A single-path test passed over the other
    for as long as the two copies existed.
    """
    # Stated, because these BCs are built bare and never attached to a grid: unstated means the
    # historical layout, and a grid is what would otherwise bind the inclusive one here.
    if uniform:
        return BoundaryConditions(
            segments=[BCSegment(name="all", bc_type=BCType.PERIODIC)],
            dimension=1,
            periodic_convention=PeriodicGridConvention.ENDPOINT_INCLUSIVE,
        )
    return BoundaryConditions(
        segments=[
            BCSegment(name="lo", bc_type=BCType.PERIODIC, boundary="x_min"),
            BCSegment(name="hi", bc_type=BCType.PERIODIC, boundary="x_max"),
        ],
        dimension=1,
        periodic_convention=PeriodicGridConvention.ENDPOINT_INCLUSIVE,
    )


@pytest.mark.parametrize("ghost_depth", [1, 2, 3])
@pytest.mark.parametrize("uniform", [True, False], ids=["uniform-path", "mixed-path"])
def test_periodic_ghosts_hold_the_analytic_continuation(ghost_depth, uniform):
    """The inclusive grid: x[0] and x[-1] coincide, so the wrap steps over the shared node."""
    x = np.linspace(*BOUNDS, NX)
    dx = x[1] - x[0]
    field = np.sin(2 * np.pi * x)
    assert abs(field[0] - field[-1]) < 1e-15, "the input must be periodic, or the ghosts prove nothing"

    bc = _periodic_bc(uniform)
    assert bc.is_uniform is uniform, "this test selects the dispatch path by is_uniform; it moved"
    assert bc.periodic_convention is PeriodicGridConvention.ENDPOINT_INCLUSIVE, (
        "this case measures the inclusive continuation; the BC default moved out from under it"
    )

    buf = PreallocatedGhostBuffer(
        interior_shape=(NX,),
        boundary_conditions=bc,
        domain_bounds=np.array([list(BOUNDS)]),
        ghost_depth=ghost_depth,
        order=2,
    )
    buf.interior[...] = field
    buf.update_ghosts()
    padded = np.asarray(buf.padded)

    g = ghost_depth
    want_low = np.sin(2 * np.pi * (x[0] - dx * np.arange(g, 0, -1)))
    want_high = np.sin(2 * np.pi * (x[-1] + dx * np.arange(1, g + 1)))

    np.testing.assert_allclose(
        padded[:g],
        want_low,
        atol=1e-14,
        err_msg=(
            f"low ghosts (depth {g}) are not the periodic continuation. Reading the last {g} "
            f"interior entries includes x[-1], which IS x[0] on an endpoint-inclusive grid, and "
            f"shifts the stencil one cell"
        ),
    )
    np.testing.assert_allclose(padded[-g:], want_high, atol=1e-14, err_msg=f"high ghosts (depth {g}) likewise")


@pytest.mark.parametrize("ghost_depth", [1, 2, 3])
def test_the_exclusive_convention_wraps_without_skipping_a_node(ghost_depth):
    """The other grid, and the reason the fill cannot hard-code either one.

    ``np.linspace(lo, hi, N, endpoint=False)`` -- what the operator layer builds -- has N distinct
    nodes, so the node one step left of ``x[0]`` is ``x[-1]`` and nothing is stepped over. This
    case and the inclusive one above differ by exactly one node, and each is the other's defect:
    filling an inclusive grid this way gave `HJBWENOSolver` a seam of 2.63e-01, and filling an
    exclusive grid the inclusive way put 0.5 of error into the operator layer's Laplacian.
    """
    n = NX
    x = np.linspace(*BOUNDS, n, endpoint=False)
    dx = x[1] - x[0]
    field = np.sin(2 * np.pi * x / (BOUNDS[1] - BOUNDS[0]))

    bc = BoundaryConditions(
        segments=[BCSegment(name="all", bc_type=BCType.PERIODIC)],
        dimension=1,
        periodic_convention=PeriodicGridConvention.ENDPOINT_EXCLUSIVE,
    )
    buf = PreallocatedGhostBuffer(
        interior_shape=(n,),
        boundary_conditions=bc,
        domain_bounds=np.array([list(BOUNDS)]),
        ghost_depth=ghost_depth,
        order=2,
    )
    buf.interior[...] = field
    buf.update_ghosts()
    padded = np.asarray(buf.padded)

    g = ghost_depth
    period = BOUNDS[1] - BOUNDS[0]
    want_low = np.sin(2 * np.pi * (x[0] - dx * np.arange(g, 0, -1)) / period)
    want_high = np.sin(2 * np.pi * (x[-1] + dx * np.arange(1, g + 1)) / period)

    np.testing.assert_allclose(padded[:g], want_low, atol=1e-14, err_msg="low ghosts, exclusive grid")
    np.testing.assert_allclose(padded[-g:], want_high, atol=1e-14, err_msg="high ghosts, exclusive grid")


@pytest.mark.parametrize("uniform", [True, False], ids=["uniform-path", "mixed-path"])
def test_a_constant_field_is_unchanged_by_the_periodic_fill(uniform):
    """Negative control: a constant has no continuation to get wrong.

    The defect above is invisible here -- every candidate source slice holds the same number -- so
    this passing while the parametrised test fails is the expected pattern, and it is the reason a
    constant or symmetric fixture cannot be the pin for this class.
    """
    buf = PreallocatedGhostBuffer(
        interior_shape=(NX,),
        boundary_conditions=_periodic_bc(uniform),
        domain_bounds=np.array([list(BOUNDS)]),
        ghost_depth=2,
        order=2,
    )
    buf.interior[...] = 3.25
    buf.update_ghosts()

    np.testing.assert_allclose(np.asarray(buf.padded), 3.25, atol=1e-15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
