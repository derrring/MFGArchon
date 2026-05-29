"""
Clipped-Voronoi nodal cells for SCNI (stabilized conforming nodal integration) of the
meshless-Galerkin operators.

For node ``x_a`` the Voronoi cell is the intersection of bisector half-planes
``{x : n_b . (x - m_ab) <= 0}`` (``n_b = x_b - x_a``, ``m_ab = (x_a + x_b)/2``), one per
Voronoi neighbour ``x_b``. Boundary cells are unbounded, so we clip to a slightly inflated
bounding box (Sutherland-Hodgman) and optionally to the domain ``Omega = {sdf <= 0}``.

SCNI needs, per cell: the area ``V_a`` and, per boundary edge ``e``, the outward unit normal
``n_e`` and length ``L_e`` (plus the edge segment, to integrate ``phi`` along it). The defining
SCNI invariant is cell CLOSURE ``sum_e n_e * L_e = 0`` (a closed polygon) -- this is exactly
what makes the smoothed nodal gradient reproduce constant gradients and conserve mass. It is
asserted per cell (fail-fast: an open polygon, e.g. from a bad sdf clip, would silently break
the integration constraint).

2D only (3D marching-cubes-style cells deferred). Issue #1139 (SCNI accuracy lever).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import Voronoi

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass
class CellGeometry:
    """A clipped Voronoi cell: CCW polygon, area, and boundary-edge normals/lengths/midpoints."""

    polygon: NDArray  # (V, 2) CCW vertices
    area: float  # |Omega_a|
    edge_midpoints: NDArray  # (E, 2)
    edge_normals: NDArray  # (E, 2) outward unit normals
    edge_lengths: NDArray  # (E,)


def _clip_halfplane(poly: NDArray, n: NDArray, c: float, eps: float = 1e-12) -> NDArray:
    """Sutherland-Hodgman: keep the part of polygon ``poly`` with ``n . x <= c``."""
    if len(poly) == 0:
        return poly
    out = []
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        da, db = float(n @ a - c), float(n @ b - c)
        a_in, b_in = da <= eps, db <= eps
        if a_in:
            out.append(a)
        if a_in != b_in:  # edge crosses the line
            t = da / (da - db)
            out.append(a + t * (b - a))
    return np.array(out) if out else np.empty((0, 2))


def _clip_sdf(poly: NDArray, sdf: Callable[[NDArray], NDArray], eps: float = 1e-12) -> NDArray:
    """Clip ``poly`` to ``{sdf <= 0}`` (boundary arc approximated by chords between crossings)."""
    s = np.asarray(sdf(poly), dtype=np.float64).ravel()
    out = []
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        sa, sb = float(s[i]), float(s[(i + 1) % m])
        a_in, b_in = sa <= eps, sb <= eps
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = sa / (sa - sb)
            out.append(a + t * (b - a))
    return np.array(out) if out else np.empty((0, 2))


def _polygon_area_ccw(poly: NDArray) -> float:
    """Signed shoelace area (positive if CCW)."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _cell_edges(poly: NDArray):
    """Outward unit normals, lengths, midpoints of a CCW polygon's edges.

    For a CCW polygon the outward normal of edge direction ``(dx, dy)`` is ``(dy, -dx)``.
    """
    v0 = poly
    v1 = np.roll(poly, -1, axis=0)
    diff = v1 - v0
    lengths = np.linalg.norm(diff, axis=1)
    keep = lengths > 1e-14
    normals = np.stack([diff[:, 1], -diff[:, 0]], axis=1)
    normals[keep] /= lengths[keep, None]
    mids = 0.5 * (v0 + v1)
    return mids[keep], normals[keep], lengths[keep]


def clipped_voronoi_cells(
    nodes: NDArray,
    bounds: list[tuple[float, float]],
    sdf: Callable[[NDArray], NDArray] | None = None,
) -> list[CellGeometry]:
    """Per-node clipped Voronoi cells for SCNI (2D).

    The cells are clipped to the rectangular domain ``bounds`` (which bounds the otherwise
    unbounded boundary cells and IS the domain when there is no ``sdf``); with ``sdf`` they are
    further clipped to ``Omega = {sdf <= 0}``. Either way the cells tile the domain, so
    ``sum_a V_a`` equals the domain area.

    Args:
        nodes: ``(N, 2)`` cloud points (must lie within ``bounds``).
        bounds: ``[(a0, b0), (a1, b1)]`` the rectangular domain / bounding box.
        sdf: optional domain SDF ``(M, 2) -> (M,)``, ``<= 0`` inside ``Omega``.

    Returns:
        ``list[CellGeometry]`` aligned with ``nodes``.

    Raises:
        ValueError: a node has an empty/degenerate cell, or a cell fails the closure invariant.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    n_nodes, dim = nodes.shape
    if dim != 2:
        raise NotImplementedError("clipped_voronoi_cells supports 2D only (SCNI 3D deferred, #1139).")
    (ax0, bx0), (ax1, bx1) = bounds
    bbox_poly = np.array([[ax0, ax1], [bx0, ax1], [bx0, bx1], [ax0, bx1]])  # CCW, the actual domain box

    vor = Voronoi(nodes, qhull_options="Qbb Qc Qz")
    neighbours: list[list[int]] = [[] for _ in range(n_nodes)]
    for i, j in vor.ridge_points:
        neighbours[i].append(j)
        neighbours[j].append(i)

    cells: list[CellGeometry] = []
    for a in range(n_nodes):
        poly = bbox_poly.copy()
        xa = nodes[a]
        for b in neighbours[a]:
            normal = nodes[b] - xa
            poly = _clip_halfplane(poly, normal, float(normal @ (0.5 * (xa + nodes[b]))))
            if len(poly) == 0:
                break
        if sdf is not None and len(poly) >= 3:
            poly = _clip_sdf(poly, sdf)

        area = _polygon_area_ccw(poly)
        if area < 0:
            poly = poly[::-1]
            area = -area
        if len(poly) < 3 or area <= 1e-14:
            raise ValueError(
                f"SCNI: node {a} at {np.round(xa, 4).tolist()} has an empty/degenerate Voronoi cell "
                f"(area={area:.2e}); check the cloud / bounds / domain (#1139)."
            )
        mids, normals, lengths = _cell_edges(poly)
        closure = float(np.abs((normals * lengths[:, None]).sum(axis=0)).max())
        if closure > 1e-9:
            raise ValueError(
                f"SCNI: node {a} cell is not closed (||sum_e n_e L_e|| = {closure:.2e}); an open "
                "polygon breaks the SCNI integration constraint (mass conservation) (#1139)."
            )
        cells.append(CellGeometry(poly, area, mids, normals, lengths))
    return cells


if __name__ == "__main__":
    """Smoke test: tiling (sum of cell areas = domain area) + closure invariant."""

    def grid(n):
        ax = np.linspace(0.0, 1.0, n)
        return np.stack([m.ravel() for m in np.meshgrid(ax, ax, indexing="ij")], axis=1)

    nodes = grid(15)
    cells = clipped_voronoi_cells(nodes, [(0.0, 1.0), (0.0, 1.0)])
    total = sum(c.area for c in cells)
    max_closure = max(float(np.abs((c.edge_normals * c.edge_lengths[:, None]).sum(axis=0)).max()) for c in cells)
    print(f"rect 15x15: N={len(cells)} sum(V_a)={total:.6f} (bbox area 1.0)  max_closure={max_closure:.2e}")
    assert abs(total - 1.0) < 1e-9, "cells must tile the bounding box"

    # disk sdf
    C, R = np.array([0.5, 0.5]), 0.4
    in_disk = lambda P: np.linalg.norm(np.atleast_2d(P) - C, axis=1) - R  # noqa: E731
    dnodes = nodes[in_disk(nodes) <= 0]
    dcells = clipped_voronoi_cells(dnodes, [(0.1, 0.9), (0.1, 0.9)], sdf=in_disk)
    darea = sum(c.area for c in dcells)
    print(f"disk:      N={len(dcells)} sum(V_a)={darea:.6f} (pi R^2 = {np.pi * R**2:.6f})")
    assert abs(darea - np.pi * R**2) < 5e-3, "disk cells must tile Omega"
    print("voronoi_cells smoke test passed.")
