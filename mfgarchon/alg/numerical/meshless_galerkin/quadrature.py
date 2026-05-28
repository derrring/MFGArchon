"""
Dimension-agnostic interior quadrature for meshless Galerkin assembly.

Tensor-product Gauss-Legendre on a background grid over the domain bounding box.
This is the dimension-agnostic "interior" rule. Boundary-clipped supports
(B(x_i, rho) intersect Omega) are the only per-dimension concern and are handled
separately (not yet implemented). For interior-dominated clouds this rule
suffices to assemble the operators and exhibits the exact Galerkin symmetry
K = K^T, which holds for any quadrature.

Issue #1131 Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def tensor_gauss(
    bounds: list[tuple[float, float]],
    n_cells: int,
    n_gauss: int,
) -> tuple[NDArray, NDArray]:
    r"""Tensor-product Gauss-Legendre quadrature over an axis-aligned box.

    Args:
        bounds: ``(a, b)`` per dimension; ``len(bounds)`` is the dimension ``d``.
        n_cells: cells per dimension (the background grid).
        n_gauss: Gauss-Legendre points per cell per dimension.

    Returns:
        ``(points, weights)`` with ``points`` of shape ``(Q, d)`` and ``weights``
        of shape ``(Q,)``, ``Q = (n_cells * n_gauss) ** d``. Exact for tensor
        polynomials of per-dimension degree ``<= 2*n_gauss - 1`` on each cell.
    """
    xi, wi = np.polynomial.legendre.leggauss(n_gauss)  # reference [-1, 1]

    per_dim_pts, per_dim_wts = [], []
    for a, b in bounds:
        edges = np.linspace(a, b, n_cells + 1)
        pts, wts = [], []
        for c in range(n_cells):
            mid = 0.5 * (edges[c] + edges[c + 1])
            half = 0.5 * (edges[c + 1] - edges[c])
            pts.append(mid + half * xi)
            wts.append(half * wi)
        per_dim_pts.append(np.concatenate(pts))
        per_dim_wts.append(np.concatenate(wts))

    pts_mesh = np.meshgrid(*per_dim_pts, indexing="ij")
    points = np.stack([m.ravel() for m in pts_mesh], axis=1)
    wts_mesh = np.meshgrid(*per_dim_wts, indexing="ij")
    weights = np.prod(np.stack([m.ravel() for m in wts_mesh], axis=1), axis=1)
    return points, weights
