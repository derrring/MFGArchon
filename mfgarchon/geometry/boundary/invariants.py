"""What each boundary condition asserts about a solved field, as a number. Issue #1574.

A solver declaring a ``BCType`` is making a claim. This module is the one place that says what
the claim MEANS operationally, so the claim can be measured instead of trusted -- and so the
measurement is the same one wherever it is taken.

It lives in the library rather than in ``tests/`` because tests are not the only consumer:
``scripts/capability_matrix.py`` drives the public solve surface, and #1574's phase 1b wants the
declaration gate itself to be able to reach these. It was previously reimplemented in at least
three test files, with the seam computed inline in each.

Two kinds of residual, and confusing them is the recurring error:

- **Exact** -- zero in exact arithmetic at every resolution, so an absolute tolerance is right.
  A periodic seam (the two entries are one physical point) and a Dirichlet wall value (it is
  pinned) are of this kind.
- **Convergent** -- zero only in the limit, so only a trend may be asserted. Mass under a no-flux
  wall and a discrete normal derivative are of this kind: the schemes here do not claim
  conservation by construction, and an absolute tolerance on them measures the grid rather than
  the solver. Measured, periodic mass drift halves per refinement (5.6e-02, 2.9e-02, 1.5e-02 at
  Nx=21/41/81); asserting ``< 1e-9`` on it reported six solvers as defective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from mfgarchon.geometry.boundary.types import BCType

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["RESIDUAL_IS_EXACT", "bc_residual", "mass_drift", "seam"]

#: Whether ``bc_residual`` for this type is zero in exact arithmetic (absolute tolerance is
#: valid) or only in the limit (only a convergence trend may be asserted). A type absent here
#: has no residual defined yet -- ``bc_residual`` raises rather than inventing one.
RESIDUAL_IS_EXACT: dict[BCType, bool] = {
    BCType.PERIODIC: True,
    BCType.DIRICHLET: True,
    BCType.NEUMANN: False,
    BCType.NO_FLUX: False,
}


def seam(field: NDArray[np.floating]) -> float:
    """``max |field[..., 0] - field[..., -1]``, the periodic identity residual.

    On an endpoint-inclusive grid the first and last entries along an axis are the same physical
    point, so this is zero for any field that respects the identification -- at every resolution,
    which is what makes an absolute tolerance valid here.
    """
    arr = np.asarray(field)
    if arr.ndim == 1:
        arr = arr[None, :]
    return float(np.abs(arr[:, 0] - arr[:, -1]).max())


def mass_drift(field: NDArray[np.floating], x: NDArray[np.floating]) -> float:
    """Relative change in total mass between the first and last time row.

    ``np.trapezoid`` is the right quadrature on an endpoint-inclusive periodic grid: the shared
    node's two half-weights sum to one full weight, so it equals the rectangle rule over the
    N-1 distinct nodes exactly when the seam is closed.
    """
    arr = np.asarray(field)
    if arr.ndim == 1:
        raise ValueError("mass_drift needs a (time, space) field; a single row has no drift")
    initial = float(np.trapezoid(arr[0], x))
    if initial == 0.0:
        raise ValueError("initial mass is zero; the relative drift would be undefined")
    return abs(float(np.trapezoid(arr[-1], x)) / initial - 1.0)


def bc_residual(
    field: NDArray[np.floating],
    bc_type: BCType,
    x: NDArray[np.floating],
    kind: Literal["HJB", "FP"],
) -> float:
    """How far ``field`` is from what ``bc_type`` asserts. Zero when the BC is honoured.

    ``kind`` selects the invariant, because the same wall means different things to the two
    equations: a no-flux wall is a statement about mass for the FP side and about the normal
    derivative of the value function for the HJB side.

    Raises:
        KeyError: for a BC type with no residual defined. Returning 0.0 for an unknown type would
            certify it, which is the failure this module exists to make impossible.
    """
    if bc_type not in RESIDUAL_IS_EXACT:
        raise KeyError(
            f"no boundary residual is defined for {bc_type.name}; add one to "
            f"{__name__}.RESIDUAL_IS_EXACT rather than treating the type as satisfied"
        )

    arr = np.asarray(field)
    arr = arr[None, :] if arr.ndim == 1 else arr

    if bc_type is BCType.PERIODIC:
        return seam(arr)
    if bc_type is BCType.DIRICHLET:
        # Homogeneous case: the wall is pinned to zero. HJB reports u at t=0, FP the final density.
        row = arr[0] if kind == "HJB" else arr[-1]
        return float(max(abs(row[0]), abs(row[-1])))
    if kind == "FP":
        return mass_drift(arr, x)  # no flux crosses the wall, so mass is conserved
    dx = float(x[1] - x[0])  # HJB: the one-sided normal derivative vanishes at each wall
    return float(max(abs(arr[0, 1] - arr[0, 0]), abs(arr[0, -1] - arr[0, -2])) / dx)
