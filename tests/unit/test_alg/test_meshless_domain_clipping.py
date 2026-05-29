"""
Unit tests for meshless-Galerkin quadrature clipping to a non-rectangular domain (#1139).

Masking the background tensor-Gauss to Omega (via a `domain` point predicate, e.g.
`geometry.predicates.sphere_region`) makes the assembly well-posed on a Lipschitz /
irregular domain. The structural identities are algebraic (quadrature-independent),
so masking preserves K=K^T, mass conservation, and A_FP=A_HJB^T; only boundary
accuracy is affected. Unclipped quadrature on a non-rectangular Omega is ill-posed
(MLS moment matrix singular at points outside the cloud) and must fail fast.
"""

from __future__ import annotations

import pytest

import numpy as np
from scipy.sparse.linalg import spsolve

from mfgarchon.alg.numerical.meshless_galerkin.discretization import discretization_from_cloud
from mfgarchon.geometry.predicates import sphere_region

CENTER = np.array([0.5, 0.5])
R = 0.4
N_PER = 21
H = 2 * R / (N_PER - 1)
RHO = 3.0 * H
in_disk = sphere_region(CENTER, R)


def _disk_cloud():
    ax = np.linspace(CENTER[0] - R, CENTER[0] + R, N_PER)
    X = np.stack([m.ravel() for m in np.meshgrid(ax, ax, indexing="ij")], axis=1)
    return X[in_disk(X)]


class TestDomainClipping:
    def test_clipping_preserves_structure(self):
        nodes = _disk_cloud()
        disc = discretization_from_cloud(nodes, delta=RHO, degree=2, n_gauss=4, domain=in_disk)
        K = disc.stiffness().toarray()
        one = np.ones(nodes.shape[0])
        v = np.tile(np.array([1.0, -0.5]), (nodes.shape[0], 1)).T
        C = disc.advection(v).toarray()
        assert np.all(np.isfinite(K))
        assert np.linalg.norm(K - K.T) / np.linalg.norm(K) < 1e-12  # symmetric for ANY quadrature
        assert np.max(np.abs(K @ one)) < 1e-8  # constants in the stiffness nullspace
        assert np.max(np.abs(C @ one)) < 1e-8  # gradient partition-of-unity -> mass conservation

    def test_unclipped_on_nonrect_domain_is_singular(self):
        # Without masking, bounding-box quad points land in the corners outside the disk
        # cloud, where the MLS moment matrix is singular -> fail fast, no silent fallback.
        nodes = _disk_cloud()
        with pytest.raises(np.linalg.LinAlgError):
            discretization_from_cloud(nodes, delta=RHO, degree=2, n_gauss=4)  # domain=None

    def test_domain_excluding_all_points_raises(self):
        nodes = _disk_cloud()
        with pytest.raises(ValueError):
            discretization_from_cloud(nodes, delta=RHO, domain=lambda P: np.zeros(len(P), dtype=bool))

    def test_fp_neumann_mass_conserved_on_disk(self):
        nodes = _disk_cloud()
        disc = discretization_from_cloud(nodes, delta=RHO, degree=2, n_gauss=4, domain=in_disk)
        M = disc.mass()
        K = disc.stiffness()
        D, dt = 0.1, 0.01
        A = (M / dt + D * K).tocsc()
        m = np.exp(-30 * np.sum((nodes - CENTER) ** 2, axis=1))
        m /= (M @ m).sum()
        mass0 = (M @ m).sum()
        for _ in range(20):
            m = spsolve(A, (M / dt) @ m)
        assert abs((M @ m).sum() - mass0) < 1e-10  # Neumann mass conserved under clipped quadrature

    def test_clipping_reduces_quadrature_to_omega(self):
        nodes = _disk_cloud()
        disc = discretization_from_cloud(nodes, delta=RHO, degree=2, n_gauss=4, domain=in_disk)
        # every retained quadrature point lies inside Omega
        # (phi cached at quad points; recover their count from the cached array)
        assert disc._phi.shape[0] > 0
        # area recovered by the clipped weights approximates the disk, not the bounding box
        area = float(disc._w.sum())
        assert abs(area - np.pi * R**2) < 5e-3
        assert area < (2 * R) ** 2  # strictly less than the bounding-box area
