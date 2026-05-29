"""
Unit tests for symmetric Nitsche Dirichlet/absorbing BC on a CURVED boundary (#1139, a2).

The boundary ∂Ω = {sdf=0} is supplied via ``BCSegment.sdf_region``; surface quadrature
(points, weights, normals) on the level set is built by marching squares
(``quadrature.surface_quadrature``) and routed into the existing Nitsche assembler.

These assert the CAPABILITY (well-posed, SPD with adequate penalty, weakly imposes g,
preserves the Type-A structure, absorbs mass) — NOT convergence: with the interim
crude-mask interior quadrature the boundary-region accuracy is floor-limited (~1e-3),
and high-order clipped quadrature (#1139 §4a) is the separate accuracy fix.

NOTE the penalty: curved 2D needs gamma >~ 50 (the flat default 20 is indefinite here);
the steady operator's coercivity is gamma > 2*C_tr, which the curved boundary Gram pushes
up. The time solve (with the M/dt shift) is more forgiving.
"""

from __future__ import annotations

import pytest

import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import spsolve

from mfgarchon.alg.numerical.meshless_galerkin.discretization import discretization_from_cloud
from mfgarchon.alg.numerical.meshless_galerkin.nitsche import _check_boundary_node_coverage, assemble_nitsche_terms
from mfgarchon.alg.numerical.meshless_galerkin.quadrature import surface_quadrature
from mfgarchon.geometry.boundary import BoundaryConditions
from mfgarchon.geometry.boundary.types import BCSegment, BCType

C = np.array([0.5, 0.5])
R = 0.4
D = 0.7
GAMMA = 100.0  # curved 2D needs gamma >~ 50 (flat default 20 is indefinite)


def disk_sdf(P):
    P = np.atleast_2d(np.asarray(P, dtype=float))
    return np.linalg.norm(P - C, axis=1) - R


def _disk_cloud(n_per=21):
    ax = np.linspace(C[0] - R, C[1] + R, n_per)
    X = np.stack([m.ravel() for m in np.meshgrid(ax, ax, indexing="ij")], axis=1)
    return X[disk_sdf(X) <= 0]


def _disk_disc(n_per=21):
    nodes = _disk_cloud(n_per)
    rho = 3.0 * (2 * R / (n_per - 1))
    disc = discretization_from_cloud(nodes, rho, degree=2, n_gauss=4, domain=lambda P: disk_sdf(P) <= 0)
    return nodes, disc


def _dirichlet_bc(g):
    return BoundaryConditions(
        segments=[BCSegment(name="bnd", bc_type=BCType.DIRICHLET, value=g, sdf_region=disk_sdf)],
        dimension=2,
    )


class TestCurvedSurfaceQuadrature:
    def test_disk_perimeter_normals_on_boundary(self):
        pts, w, n = surface_quadrature(disk_sdf, [(0.08, 0.92), (0.08, 0.92)], 48)
        assert abs(w.sum() - 2 * np.pi * R) < 1e-2  # arc-length weights recover the perimeter
        assert np.max(np.abs(disk_sdf(pts))) < 1e-3  # points lie on {sdf=0}
        assert np.max(np.abs(np.linalg.norm(n, axis=1) - 1.0)) < 1e-10  # unit
        radial = (pts - C) / np.linalg.norm(pts - C, axis=1, keepdims=True)
        assert np.min(np.sum(n * radial, axis=1)) > 0.99  # outward (= radial for a disk)

    def test_3d_not_implemented(self):
        with pytest.raises(NotImplementedError):
            surface_quadrature(lambda P: np.linalg.norm(P, axis=1) - 1.0, [(0.0, 1.0)] * 3, 8)

    def test_no_crossing_raises(self):
        with pytest.raises(ValueError):
            surface_quadrature(lambda P: np.linalg.norm(np.atleast_2d(P) - C, axis=1) - 5.0, [(0.0, 1.0)] * 2, 16)


class TestCurvedNitsche:
    def test_spd_and_imposes_g(self):
        # harmonic u = (x-cx)^2 - (y-cy)^2 (Laplace u = 0 -> f = 0), Dirichlet g = u on ∂Ω
        nodes, disc = _disk_disc()
        g = lambda x: float((x[0] - C[0]) ** 2 - (x[1] - C[1]) ** 2)  # noqa: E731
        N, rhs = assemble_nitsche_terms(disc, _dirichlet_bc(g), D, GAMMA, 4, include_data=True)
        A = (D * disc.stiffness() + N).tocsr()
        assert eigvalsh(0.5 * (A.toarray() + A.toarray().T)).min() > 0  # SPD at adequate penalty
        U = spsolve(A, rhs)
        u_ex = (nodes[:, 0] - C[0]) ** 2 - (nodes[:, 1] - C[1]) ** 2
        assert np.sqrt(np.mean((U - u_ex) ** 2)) < 3e-2  # quadrature-floor-limited, not converging
        # weak boundary trace tracks g
        pts, _w, n = surface_quadrature(disk_sdf, [(0.08, 0.92), (0.08, 0.92)], 48)
        phi_b, _gn = disc.boundary_shape_data(pts, n)
        g_b = np.array([g(p) for p in pts])
        assert np.max(np.abs(phi_b @ U - g_b)) < 6e-2

    def test_block_symmetric_and_dual(self):
        _nodes, disc = _disk_disc()
        bc = _dirichlet_bc(0.0)
        N_hjb, _ = assemble_nitsche_terms(disc, bc, D, GAMMA, 4, include_data=True)
        N_fp, rhs_fp = assemble_nitsche_terms(disc, bc, D, GAMMA, 4, include_data=False)
        assert abs(N_hjb - N_hjb.T).max() < 1e-10  # symmetric
        assert abs(N_hjb - N_fp).max() == 0.0  # FP block identical -> A_FP = A_HJB^T
        assert rhs_fp is None

    def test_absorbing_fp_loses_mass(self):
        nodes, disc = _disk_disc()
        N, _ = assemble_nitsche_terms(disc, _dirichlet_bc(0.0), D, GAMMA, 4, include_data=False)
        M, K = disc.mass(), disc.stiffness()
        dt = 0.005
        A = (M / dt + D * K + N).tocsr()
        m = np.exp(-40 * np.sum((nodes - C) ** 2, axis=1))
        m /= (M @ m).sum()
        mass0 = (M @ m).sum()
        for _ in range(25):
            m = np.maximum(spsolve(A, (M / dt) @ m), 0.0)
        mass = (M @ m).sum()
        assert np.all(np.isfinite(m))
        assert mass < 0.99 * mass0  # mass leaves through the absorbing curved boundary
        assert mass <= mass0 + 1e-9

    def test_uncovered_boundary_fails_fast(self):
        # boundary points with no cloud node within rho -> greppable error, not deep LinAlgError
        _nodes, disc = _disk_disc()
        far = np.array([[5.0, 5.0], [6.0, 6.0]])
        with pytest.raises(ValueError, match="cloud must cover"):
            _check_boundary_node_coverage(far, disc)
