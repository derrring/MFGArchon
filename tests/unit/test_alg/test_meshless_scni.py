"""
Unit tests for the SCNI (stabilized conforming nodal integration) meshless discretization (#1139).

The decisive property is BC-free: plain SCNI is LINEARLY consistent, so the smoothed nodal
gradient reproduces constant gradients EXACTLY (`grad~(a + b.x) = b`) — unlike Gauss quadrature
of the rational MLS integrands, whose patch error floors at ~5e-2. From `grad~ 1 = 0` (partition
of unity + per-cell closure) follow `K@1 = 0` and `advection@1 = 0` (mass conservation). These
need no Dirichlet solve, so they don't depend on the Nitsche chain (#1140-#1142).

(Quadratic-exactness needs NSNI — a follow-up; not tested here.)
"""

from __future__ import annotations

import numpy as np

from mfgarchon.alg.numerical.meshless_galerkin.scni_discretization import MeshlessSCNIDiscretization
from mfgarchon.alg.numerical.weak_form_discretization import WeakFormDiscretization

BOUNDS = [(0.0, 1.0), (0.0, 1.0)]


def _grid(n):
    ax = np.linspace(0.0, 1.0, n)
    return np.stack([m.ravel() for m in np.meshgrid(ax, ax, indexing="ij")], axis=1)


def _disc(n=15):
    nodes = _grid(n)
    return nodes, MeshlessSCNIDiscretization(nodes, rho=3.0 / (n - 1), degree=2, bounds=BOUNDS)


class TestSCNIDiscretization:
    def test_protocol_conformance(self):
        _nodes, disc = _disc(11)
        assert isinstance(disc, WeakFormDiscretization)

    def test_smoothed_gradient_reproduces_linear(self):
        # DECISIVE: grad~ reproduces constant gradients exactly (the Gauss version floors ~5e-2).
        nodes, disc = _disc(15)
        B0, B1 = disc.smoothed_gradient()
        one = np.ones(len(nodes))
        x, y = nodes[:, 0], nodes[:, 1]
        assert np.max(np.abs(B0 @ one)) < 1e-10  # grad~ of a constant = 0
        assert np.max(np.abs(B1 @ one)) < 1e-10
        assert np.max(np.abs(B0 @ x - 1.0)) < 1e-9  # d/dx (x) = 1
        assert np.max(np.abs(B0 @ y)) < 1e-9  # d/dx (y) = 0
        assert np.max(np.abs(B1 @ x)) < 1e-9
        assert np.max(np.abs(B1 @ y - 1.0)) < 1e-9

    def test_stiffness_symmetric_and_constant_nullspace(self):
        nodes, disc = _disc(15)
        K = disc.stiffness().toarray()
        assert np.linalg.norm(K - K.T) / max(np.linalg.norm(K), 1e-30) < 1e-12
        assert np.max(np.abs(K @ np.ones(len(nodes)))) < 1e-9  # constants in the nullspace

    def test_mass_lumped_tiles_domain(self):
        _nodes, disc = _disc(15)
        M = disc.mass()
        assert M.nnz == disc.n_dof  # diagonal (lumped)
        assert abs(float(M.sum()) - 1.0) < 1e-9  # nodal volumes tile [0,1]^2

    def test_advection_conserves_mass(self):
        nodes, disc = _disc(15)
        v = np.tile(np.array([0.7, -0.3]), (len(nodes), 1)).T
        C = disc.advection(v).toarray()
        # C@1 = 0 (from grad~ 1 = 0) => FP operator -C^T has zero column sums => mass conserved
        assert np.max(np.abs(C @ np.ones(len(nodes)))) < 1e-9

    def test_lloyd_cloud_linear_reproduction(self):
        # SCNI on a realistic scattered (Lloyd/CVT) cloud, not a structured grid.
        from mfgarchon.geometry.collocation import ImplicitDomainCollocation
        from mfgarchon.geometry.implicit import Hyperrectangle

        nodes = ImplicitDomainCollocation(Hyperrectangle(bounds=BOUNDS)).sample_interior(220, method="lloyd", seed=0)
        h = 1.0 / np.sqrt(len(nodes))
        disc = MeshlessSCNIDiscretization(nodes, rho=3.0 * h, degree=2, bounds=BOUNDS)
        B0, B1 = disc.smoothed_gradient()
        err = np.abs(B0 @ nodes[:, 0] - 1.0)
        # Linear reproduction is machine-exact at well-conditioned nodes (median ~1e-13); a few
        # near-boundary nodes degrade to ~1e-4 due to ill-conditioned MLS moment matrices on a
        # scattered cloud (the cloud-quality effect -- Lloyd >> Poisson, but still not a grid).
        assert np.median(err) < 1e-9
        assert err.max() < 1e-3
        assert np.max(np.abs(B1 @ np.ones(len(nodes)))) < 1e-8  # closure ~machine, tiny boundary outlier

    def test_lipschitz_disk_sdf_clip(self):
        # SCNI on a non-rectangular (disk) domain via the sdf-clip: linear-exact + conservative,
        # i.e. the core holds on the actual Lipschitz target (BC-free; Nitsche is separate).
        C, R = np.array([0.5, 0.5]), 0.4
        sdf = lambda P: np.linalg.norm(np.atleast_2d(P) - C, axis=1) - R  # noqa: E731
        ax = np.linspace(0.1, 0.9, 19)
        grid = np.stack([m.ravel() for m in np.meshgrid(ax, ax, indexing="ij")], axis=1)
        nodes = grid[sdf(grid) <= 0]
        disc = MeshlessSCNIDiscretization(nodes, rho=3.0 * 0.8 / 18, degree=2, bounds=[(0.1, 0.9), (0.1, 0.9)], sdf=sdf)
        B0, _B1 = disc.smoothed_gradient()
        one = np.ones(len(nodes))
        err = np.abs(B0 @ nodes[:, 0] - 1.0)
        assert np.median(err) < 1e-9  # linear reproduction holds on the curved (sdf-clipped) domain
        assert err.max() < 1e-6
        assert np.max(np.abs(B0 @ one)) < 1e-8
        assert abs(float(disc.mass().sum()) - np.pi * R**2) < 5e-3  # cells tile the disk (chord approx)
        K = disc.stiffness().toarray()
        assert np.linalg.norm(K - K.T) / np.linalg.norm(K) < 1e-12
        assert np.max(np.abs(K @ one)) < 1e-8
