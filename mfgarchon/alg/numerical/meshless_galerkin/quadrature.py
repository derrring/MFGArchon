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
    from collections.abc import Callable

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


def surface_quadrature(
    sdf: Callable[[NDArray], NDArray],
    bounds: list[tuple[float, float]],
    n_cells: int,
) -> tuple[NDArray, NDArray, NDArray]:
    r"""Surface quadrature on the curved boundary ``{x : sdf(x) = 0}`` of a domain.

    Midpoint rule on the zero level set extracted by marching squares (2D) over an
    ``n_cells`` background grid on ``bounds``; the boundary measure is the segment
    length (1 in 1D, where a boundary is a point), and outward unit normals come from
    ``outward_normal_from_sdf`` (the SDF convention is ``sdf < 0`` inside, so the
    gradient points outward -- no sign flip). This is the curved-boundary source for
    the symmetric Nitsche terms (#1139), the analogue of ``boundary_tensor_gauss`` for
    axis-aligned faces.

    Args:
        sdf: signed distance / level-set function, ``(N, d) -> (N,)``, negative inside.
        bounds: ``(a, b)`` per dimension; ``len(bounds)`` is the dimension ``d``.
        n_cells: background cells per dimension for the marching grid.

    Returns:
        ``(points, weights, normals)`` of shapes ``(Q, d)``, ``(Q,)``, ``(Q, d)``.

    Raises:
        NotImplementedError: ``d >= 3`` (marching-cubes deferred; #1139).
        ValueError: no zero crossing found within ``bounds``.
    """
    d = len(bounds)
    if d == 1:
        return _surface_quadrature_1d(sdf, bounds, n_cells)
    if d == 2:
        return _surface_quadrature_2d(sdf, bounds, n_cells)
    raise NotImplementedError("surface_quadrature: 3D marching-cubes deferred (#1139); only 1D/2D supported.")


def _surface_quadrature_1d(sdf, bounds, n_cells):
    a, b = bounds[0]
    xs = np.linspace(a, b, n_cells + 1)
    phi = np.asarray(sdf(xs[:, None]), dtype=np.float64).ravel()
    pts, normals = [], []
    for i in range(n_cells):
        s0, s1 = phi[i], phi[i + 1]
        if (s0 <= 0) != (s1 <= 0):  # boundary crossing on this interval
            t = s0 / (s0 - s1)
            pts.append([xs[i] + t * (xs[i + 1] - xs[i])])
            normals.append([1.0 if s1 > s0 else -1.0])  # outward = sign of sdf'
    if not pts:
        raise ValueError("surface_quadrature(1D): sdf has no zero crossing in bounds; check sdf/bounds.")
    pts = np.asarray(pts, dtype=np.float64)
    return pts, np.ones(pts.shape[0]), np.asarray(normals, dtype=np.float64)


def _surface_quadrature_2d(sdf, bounds, n_cells):
    from mfgarchon.operators.differential.function_gradient import outward_normal_from_sdf

    (ax0, bx0), (ax1, bx1) = bounds
    xs = np.linspace(ax0, bx0, n_cells + 1)
    ys = np.linspace(ax1, bx1, n_cells + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    phi = np.asarray(sdf(np.stack([gx.ravel(), gy.ravel()], axis=1)), dtype=np.float64).reshape(
        n_cells + 1, n_cells + 1
    )

    points, weights = [], []
    for i in range(n_cells):
        for j in range(n_cells):
            # cell corners CCW: bottom-left, bottom-right, top-right, top-left
            corners = [
                (xs[i], ys[j], phi[i, j]),
                (xs[i + 1], ys[j], phi[i + 1, j]),
                (xs[i + 1], ys[j + 1], phi[i + 1, j + 1]),
                (xs[i], ys[j + 1], phi[i, j + 1]),
            ]
            crossings = []
            for k in range(4):
                x0, y0, s0 = corners[k]
                x1, y1, s1 = corners[(k + 1) % 4]
                if (s0 <= 0) != (s1 <= 0):  # zero crossing on this edge (phi=0 -> inside)
                    t = s0 / (s0 - s1)
                    crossings.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
            # 2 crossings -> one segment; 4 (saddle, rare for smooth SDF) -> pair in edge
            # order (a tiny arc-length error if mispaired, negligible for quadrature).
            if len(crossings) == 2:
                segments = [(crossings[0], crossings[1])]
            elif len(crossings) == 4:
                segments = [(crossings[0], crossings[1]), (crossings[2], crossings[3])]
            else:
                continue
            for p, q in segments:
                p, q = np.asarray(p), np.asarray(q)
                length = float(np.linalg.norm(q - p))
                if length > 0.0:
                    points.append(0.5 * (p + q))
                    weights.append(length)

    if not points:
        raise ValueError("surface_quadrature(2D): sdf has no zero level set in bounds; check sdf/bounds.")
    points = np.asarray(points, dtype=np.float64)
    normals = outward_normal_from_sdf(sdf, points)
    return points, np.asarray(weights, dtype=np.float64), np.asarray(normals, dtype=np.float64)
