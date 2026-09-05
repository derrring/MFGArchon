"""The measure on a node set. One owner, depending on nothing but numpy.

Lives here rather than on the grid because both directions need it and only one of them can own an
import: ``geometry.collocation`` imports ``utils.numerical``, so ``utils.numerical`` cannot import
``geometry``. A leaf module both can reach is the only place a shared primitive fits.

See Issue #2145 for the decision this implements and the census of the 169 places that answered it
independently before there was an owner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = ["quadrature_weights_1d", "quadrature_weights_nd"]


def quadrature_weights_1d(coordinates: NDArray) -> NDArray:
    """The control volume each node owns along one axis.

        w[0] = (x[1] - x[0]) / 2 ;  w[i] = (x[i+1] - x[i-1]) / 2 ;  w[-1] = (x[-1] - x[-2]) / 2

    On an ENDPOINT-INCLUSIVE node set -- what ``TensorProductGrid`` builds, where the boundary lies
    ON a node -- the two end nodes own HALF a cell each. Those are the trapezoid weights, exactly.

    ``sum(f) * dx`` is a different functional: it gives each end node a full cell reaching outside
    the declared bounds and over-counts by ``dx*(f[0]+f[-1])/2`` -- 3.5% on a standard fixture,
    before anything evolves. The same statement holds one dimension down, so a boundary FACE uses
    these weights along the axes it spans.

    Written on coordinates rather than on a single ``dx``, so a graded axis is not a special case.
    """
    x = np.asarray(coordinates, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"coordinates must be 1-D; got shape {x.shape}")
    if x.size < 2:
        raise ValueError(f"axis has {x.size} node(s); a measure needs at least two")
    w = np.empty_like(x)
    w[0] = (x[1] - x[0]) / 2.0
    w[-1] = (x[-1] - x[-2]) / 2.0
    w[1:-1] = (x[2:] - x[:-2]) / 2.0
    return w


def quadrature_weights_nd(coordinates: Sequence[NDArray]) -> NDArray:
    """The control volume each node owns on a tensor-product grid: the outer product of the axes.

    A corner node owns ``prod(h_d / 2)``, an edge node one half-factor per axis it is a wall of,
    an interior node ``prod(h_d)``. Shape is ``tuple(len(c) for c in coordinates)``, so it
    broadcasts against a field laid out like the grid, and ``(w * field).sum()`` is the integral.

    ``d = 1`` is not a special case -- it returns `quadrature_weights_1d` of the single axis. That
    matters: a caller that reaches for a 1-D helper and an nD helper has two implementations of one
    convention, which is the defect #2237 and #2243 were about. There is one here.
    """
    axes = list(coordinates)
    if not axes:
        raise ValueError("quadrature_weights_nd needs at least one axis")
    weights = quadrature_weights_1d(axes[0])
    for axis in axes[1:]:
        weights = np.multiply.outer(weights, quadrature_weights_1d(axis))
    return weights
