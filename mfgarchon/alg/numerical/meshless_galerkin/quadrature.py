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


def boundary_tensor_gauss(
    bounds: list[tuple[float, float]],
    faces: list[tuple[int, str]],
    n_cells: int = 1,
    n_gauss: int = 4,
) -> tuple[NDArray, NDArray, NDArray]:
    r"""Surface quadrature on selected faces of an axis-aligned bounding box.

    Each face ``(axis, side)`` fixes one coordinate at the box bound and tensors a
    Gauss-Legendre rule over the remaining ``d - 1`` axes (a single unit-weight
    point in 1D, where a face is a point). This is the interim Dirichlet-boundary
    rule for the meshless Nitsche terms: full ``B(x_i, rho) intersect Omega``
    clipping is tracked under #1139.

    Args:
        bounds: ``(a, b)`` per dimension; ``len(bounds)`` is the dimension ``d``.
        faces: ``(axis, side)`` pairs, ``side in {"min", "max"}``.
        n_cells: background cells per free dimension.
        n_gauss: Gauss-Legendre points per cell per free dimension.

    Returns:
        ``(points, weights, normals)`` of shapes ``(Q, d)``, ``(Q,)``, ``(Q, d)``.
        ``normals`` are the outward unit normals ``+/- e_axis`` per face.
    """
    d = len(bounds)
    all_pts: list[NDArray] = []
    all_wts: list[NDArray] = []
    all_nrm: list[NDArray] = []
    for axis, side in faces:
        if side not in ("min", "max"):
            raise ValueError(f"face side must be 'min' or 'max', got {side!r}")
        fixed = bounds[axis][0] if side == "min" else bounds[axis][1]
        normal = np.zeros(d)
        normal[axis] = -1.0 if side == "min" else 1.0

        free_axes = [k for k in range(d) if k != axis]
        if not free_axes:  # 1D: the face is a single point, surface measure 1
            pts = np.array([[fixed]], dtype=np.float64)
            wts = np.array([1.0])
        else:
            fpts, wts = tensor_gauss([bounds[k] for k in free_axes], n_cells, n_gauss)
            pts = np.empty((fpts.shape[0], d), dtype=np.float64)
            pts[:, axis] = fixed
            for j, k in enumerate(free_axes):
                pts[:, k] = fpts[:, j]
        all_pts.append(pts)
        all_wts.append(wts)
        all_nrm.append(np.tile(normal, (pts.shape[0], 1)))

    return np.vstack(all_pts), np.concatenate(all_wts), np.vstack(all_nrm)
