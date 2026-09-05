"""
Unit tests for the meshless-Galerkin symmetric Nitsche Dirichlet assembly (#1138).

Operator-level (no MFGProblem): boundary quadrature on bounding-box faces, the
Nitsche block ``-D*B - D*B^T + (gamma*D/rho)*P``, a manufactured Dirichlet Poisson
solve (convergence + inhomogeneous-data path), SPD/symmetry, and the HJB/FP block
identity that underpins the Type-A transpose duality ``A_FP = A_HJB^T``.
"""

from __future__ import annotations

import pytest

import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import spsolve

from mfgarchon.alg.numerical.meshless_galerkin.discretization import (
    MeshlessGalerkinDiscretization,
    discretization_from_cloud,
)
from mfgarchon.alg.numerical.meshless_galerkin.mls_basis import shape_functions_and_grads
from mfgarchon.alg.numerical.meshless_galerkin.nitsche import (
    _domain_bounds,
    _segment_quadrature,
    assemble_nitsche_terms,
    dirichlet_segments,
)
from mfgarchon.alg.numerical.meshless_galerkin.quadrature import boundary_tensor_gauss
from mfgarchon.geometry.boundary import BoundaryConditions
from mfgarchon.geometry.boundary.types import BCSegment, BCType

D = 0.7  # nontrivial diffusion: confirms every Nitsche term scales with D


def _dirichlet_bc(values: dict[str, float], dim: int = 1) -> BoundaryConditions:
    return BoundaryConditions(
        segments=[BCSegment(name=face, bc_type=BCType.DIRICHLET, value=v, boundary=face) for face, v in values.items()],
        dimension=dim,
    )


def _poisson_1d(N: int, u_exact, f_func, degree: int = 2, gamma: float = 20.0):
    """Steady ``-D u'' = f`` on [0,1] with Dirichlet BC via Nitsche; returns (err, A, N_block)."""
    nodes = np.linspace(0.0, 1.0, N)[:, None]
    disc = discretization_from_cloud(nodes, delta=3.5 / (N - 1), degree=degree, n_gauss=6)
    K, M = disc.stiffness(), disc.mass()
    bc = _dirichlet_bc({"x_min": float(u_exact(np.array([0.0]))[0]), "x_max": float(u_exact(np.array([1.0]))[0])})
    N_block, rhs_data = assemble_nitsche_terms(disc, bc, D, gamma, n_gauss=6, include_data=True)
    A = (D * K + N_block).tocsr()
    rhs = M @ f_func(nodes[:, 0])
    if rhs_data is not None:
        rhs = rhs + rhs_data
    U = spsolve(A, rhs)
    # MLS is non-interpolatory: the coefficients U_j are NOT u(x_j). The solution error
    # is ||u_h - u_exact|| with u_h(x_i) = sum_j phi_j(x_i) U_j, NOT ||U - u_exact(nodes)||
    # (the latter measures the coefficient-vs-value gap, a different quantity).
    phi_nodes, _ = shape_functions_and_grads(nodes, nodes, disc._rho, disc._exps, "numpy")
    u_h = phi_nodes @ U
    err = np.sqrt(np.mean((u_h - u_exact(nodes[:, 0])) ** 2))
    return err, A, N_block


class TestBoundaryQuadrature:
    def test_1d_faces_points_weights_normals(self):
        x, w, n = boundary_tensor_gauss([(0.0, 1.0)], [(0, "min"), (0, "max")], n_gauss=4)
        assert np.allclose(x.ravel(), [0.0, 1.0])
        assert np.allclose(w, [1.0, 1.0])  # 0-d face has unit surface measure
        assert np.allclose(n.ravel(), [-1.0, 1.0])

    def test_2d_edge_length_and_normal(self):
        x, w, n = boundary_tensor_gauss([(0.0, 1.0), (0.0, 2.0)], [(0, "min")], n_gauss=4)
        assert abs(w.sum() - 2.0) < 1e-12  # |x_min edge| = 2
        assert np.allclose(x[:, 0], 0.0)
        assert np.allclose(n, np.tile([-1.0, 0.0], (len(w), 1)))

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            boundary_tensor_gauss([(0.0, 1.0)], [(0, "middle")])


class TestNitscheAssembly:
    def test_no_dirichlet_returns_none(self):
        from mfgarchon.geometry.boundary import no_flux_bc

        disc = discretization_from_cloud(np.linspace(0, 1, 21)[:, None], 3.5 / 20, degree=2, n_gauss=4)
        N_block, rhs = assemble_nitsche_terms(disc, no_flux_bc(dimension=1), D, 20.0, 4, include_data=True)
        assert N_block is None
        assert rhs is None

    def test_block_symmetric(self):
        disc = discretization_from_cloud(np.linspace(0, 1, 41)[:, None], 3.5 / 40, degree=2, n_gauss=6)
        bc = _dirichlet_bc({"x_min": 0.0, "x_max": 0.0})
        N_block, _ = assemble_nitsche_terms(disc, bc, D, 20.0, 6, include_data=True)
        assert abs(N_block - N_block.T).max() < 1e-10

    def test_augmented_operator_spd(self):
        _, A, _ = _poisson_1d(81, lambda x: np.sin(np.pi * x), lambda x: D * np.pi**2 * np.sin(np.pi * x))
        Adense = A.toarray()
        assert np.abs(Adense - Adense.T).max() < 1e-10
        assert eigvalsh(0.5 * (Adense + Adense.T)).min() > 0.0  # Dirichlet removes the constant nullspace

    def test_hjb_fp_block_identical(self):
        """The symmetric block is identical for HJB (data) and FP (no data): A_FP = A_HJB^T."""
        disc = discretization_from_cloud(np.linspace(0, 1, 41)[:, None], 3.5 / 40, degree=2, n_gauss=6)
        bc = _dirichlet_bc({"x_min": 0.0})
        N_hjb, _ = assemble_nitsche_terms(disc, bc, D, 20.0, 6, include_data=True)
        N_fp, rhs_fp = assemble_nitsche_terms(disc, bc, D, 20.0, 6, include_data=False)
        assert abs(N_hjb - N_fp).max() == 0.0
        assert rhs_fp is None


class TestManufacturedConvergence:
    def test_homogeneous_dirichlet_eoc(self):
        """u(x)=sin(pi x), g=0: solution error (reconstructed u_h) converges, but the rate
        degrades toward a QUADRATURE FLOOR -- Gauss quadrature of the rational MLS integrands
        is inexact, so the observed EOC drops to ~1.4-1.5 at fine h rather than the degree-2
        optimum 2. Lifting this needs stabilized nodal integration (SCNI), not more Gauss
        points (which converge only ~1/n_gauss). See the MLS-quadrature diagnostic."""
        errs = [
            _poisson_1d(N, lambda x: np.sin(np.pi * x), lambda x: D * np.pi**2 * np.sin(np.pi * x))[0]
            for N in (21, 41, 81, 161)
        ]
        rates = [np.log(errs[i - 1] / errs[i]) / np.log(2) for i in range(1, len(errs))]
        assert errs[0] / errs[-1] > 10.0  # converges by >1 order over the refinement
        assert min(rates) > 1.2, f"convergence stalled below the quadrature floor: {rates}"

    def test_linear_reproduction_inhomogeneous_g(self):
        """u(x)=1+2x, f=0, g=(1,3): exercises the f_sym + f_pen data path; reproduced ~exactly."""
        err, _, _ = _poisson_1d(101, lambda x: 1.0 + 2.0 * x, lambda x: np.zeros_like(x))
        assert err < 1e-5


def _poisson_2d(n: int, gamma: float = 100.0, degree: int = 2):
    """Manufactured ``-D lap(u) = f`` on the unit square, ``u = sin(pi x) sin(pi y)``.

    Homogeneous Dirichlet imposed weakly on all four faces; structured ``n x n`` cloud,
    ``rho = 3.5 h``. Returns the RMS error of the RECONSTRUCTED field ``phi @ U``.
    """
    ax = np.linspace(0.0, 1.0, n)
    nodes = np.stack([m.ravel() for m in np.meshgrid(ax, ax, indexing="ij")], axis=1)
    disc = discretization_from_cloud(nodes, delta=3.5 / (n - 1), degree=degree, n_gauss=6)
    K, M = disc.stiffness(), disc.mass()
    bc = _dirichlet_bc(dict.fromkeys(("x_min", "x_max", "y_min", "y_max"), 0.0), dim=2)
    N_block, _ = assemble_nitsche_terms(disc, bc, D, gamma, n_gauss=6, include_data=True)
    f = D * 2.0 * np.pi**2 * np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
    U = spsolve((D * K + N_block).tocsr(), M @ f)
    phi_nodes, _ = shape_functions_and_grads(nodes, nodes, disc._rho, disc._exps, "numpy")
    u_exact = np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
    return float(np.sqrt(np.mean((phi_nodes @ U - u_exact) ** 2)))


class TestBoundaryRuleResolvesTheSupportScale:
    """#1679: the Dirichlet boundary rule must refine with the cloud, not with the domain.

    The Nitsche boundary integrand is ``phi_i phi_j`` and ``phi_i (n . grad phi_j)``, which
    varies on the MLS support scale ``rho``. ``rho`` shrinks under refinement, so a rule
    whose resolution does not follow it resolves strictly less of the integrand at every
    level. Before the fix, ``_segment_quadrature`` took ``boundary_tensor_gauss``'s
    ``n_cells=1`` default: the 2-D rule stayed at 24 points from n=11 to n=26 while the
    volume rule grew 3600 -> 22500, and the manufactured 2-D EOC ran -0.49 / -0.93 / -0.84.

    ``d >= 2`` IS REQUIRED and is not a preference. ``quadrature.py`` builds a 1-D face as a
    single point of unit surface measure, on a branch that never reads ``n_cells``, so a 1-D
    fixture cannot fail either assertion however the rule is sized -- which is why the 1-D
    EOC test above stayed green through the whole defect.
    """

    LEVELS = (7, 11, 16, 21, 26)
    N_GAUSS = 6
    N_FACES = 4  # the unit square's four bounding-box faces
    MAX_SIDE = 1.0

    def _boundary_points(self, n):
        """``(Q_b, rho)`` from the SHIPPED path, with the volume rule stubbed out.

        ``_segment_quadrature`` reads only ``rho``/``dim``/``dof_coordinates``, so the volume
        quadrature is replaced by a 1-point dummy: the assertion below is about the boundary
        rule, and building the real volume rule costs ~5 s at n=26 for something never read.
        """
        ax = np.linspace(0.0, 1.0, n)
        nodes = np.stack([m.ravel() for m in np.meshgrid(ax, ax, indexing="ij")], axis=1)
        rho = 3.5 / (n - 1)
        disc = MeshlessGalerkinDiscretization(nodes, rho, 2, np.array([[0.5, 0.5]]), np.array([1.0]), backend="numpy")
        bc = _dirichlet_bc(dict.fromkeys(("x_min", "x_max", "y_min", "y_max"), 0.0), dim=2)
        bounds = _domain_bounds(disc)
        q_b = sum(len(_segment_quadrature(s, disc, bounds, self.N_GAUSS)[0]) for s in dirichlet_segments(bc))
        return q_b, rho

    def test_the_rule_holds_two_cells_per_support_radius_at_every_level(self):
        """The contract, asserted directly -- NOT the raw point count.

        A growing count is not the property: a rule can add points while still resolving
        *less* of the support scale. Measured: with the coefficient halved the counts go
        [24, 48, 72, 72] -- monotone, so a count-only pin passes -- while points per support
        radius decay 4.20 -> 3.15 and the manufactured EOC reaches -9.19 at n=21.

        So assert what the formula promises: ``n_cells >= 2 * max_side / rho``, i.e. at least
        two cells, hence ``2 * n_gauss`` quadrature points, within every support radius along
        a face.

        This pins the CONTRACT, not convergence, and deliberately so: measured, it fires on
        the pre-#1679 rule (2.10 at n=11 down to 0.84 at n=26, against a floor of 12) and on
        coefficients 0.5, 1.0 and 1.5, passing only at the shipped 2.0. Coefficient 1.0 does
        still converge, so this is stricter than "it converges" -- weakening the constant is
        exactly the decision that should have to trip a test and be re-argued. Retirement
        condition: it trips whenever the rule stops delivering two cells per support radius.
        """
        floor = 2 * self.N_GAUSS
        observed = {}
        for n in self.LEVELS:
            q_b, rho = self._boundary_points(n)
            observed[n] = (q_b / self.N_FACES) * rho / self.MAX_SIDE
        worst = min(observed.values())
        assert worst >= floor, (
            f"boundary rule resolves less than {floor} points per support radius: {observed} "
            f"(pre-#1679 this ran 2.10 at n=11 down to 0.84 at n=26)"
        )

    def test_manufactured_2d_dirichlet_converges(self):
        """External oracle: ``u = sin(pi x) sin(pi y)`` is exact and computed independently.

        Measured at the fix: rate +3.13 over n = 7 -> 11. Pinned at 1.5 because the claim is
        that the scheme CONVERGES, not that it attains any particular order. With the boundary
        rule pinned at one cell the same two levels give -4.24, so this cannot pass by accident.

        What this does NOT catch on its own: a rule that is merely too coarse rather than
        unscaled still converges over 7 -> 11 and only turns at n=21, which costs 16 s. The
        assertion above is what covers that case, and it costs nothing.
        """
        errs = [_poisson_2d(n) for n in (7, 11)]
        rate = np.log(errs[0] / errs[1]) / np.log((1 / 6) / (1 / 10))
        assert rate > 1.5, f"2-D Nitsche does not converge under refinement (#1679): rate {rate:+.2f}, errs {errs}"
