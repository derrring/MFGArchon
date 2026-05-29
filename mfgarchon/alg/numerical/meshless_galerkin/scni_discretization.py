"""
SCNI (stabilized conforming nodal integration) discretization for the meshless-Galerkin
weak form -- an alternative to the Gauss-quadrature ``MeshlessGalerkinDiscretization`` that
lifts the accuracy floor caused by inexact Gauss quadrature of the rational MLS integrands.

Per node ``x_a`` with clipped Voronoi cell ``Omega_a`` (area ``V_a``, boundary edges ``e`` with
outward unit normal ``n_e``), the SMOOTHED nodal gradient is the divergence-theorem identity

    grad~_d phi_j(x_a) = (1/V_a) * sum_e n_{e,d} * integral_e phi_j ds,

the edge integral by a short Gauss-Legendre rule. The operators are then assembled by a single
nodal sum (no interior quadrature of the rational integrand):

- stiffness   K = sum_d B~_d^T diag(V) B~_d                       (symmetric)
- mass        M = diag(V)                                         (lumped; sum = |Omega|)
- gradient_projection  R_d = diag(V) B~_d   so the solver's G_d = diag(1/M_lumped) R_d = B~_d
- advection   C[i,j] = sum_a V_a phi_i(x_a) (alpha(x_a) . B~[a,j])  (only the TRIAL phi_j smoothed)

Mass conservation / `K@1=0` follow from the per-cell CLOSURE invariant ``sum_e n_e L_e = 0``
(asserted in ``voronoi_cells``) plus MLS reproduction of constants; the edge-integral PoU then
makes ``sum_j B~_d[a,j] = 0`` exactly, independent of the edge-Gauss order.

Plain SCNI is LINEARLY consistent (reproduces constant gradients exactly); quadratic-exactness
needs NSNI (an extra moment) -- a follow-up. 2D only. Reuses ``SchemeFamily.MESHLESS_GALERKIN``
(Type-A discrete-dual; ``K`` symmetric + FP advection = ``-C^T``). Issue #1139.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from mfgarchon.alg.numerical.meshless_galerkin.mls_basis import monomial_exponents, shape_functions_and_grads
from mfgarchon.alg.numerical.meshless_galerkin.voronoi_cells import clipped_voronoi_cells

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class MeshlessSCNIDiscretization:
    """Point cloud + MLS + SCNI implementation of ``WeakFormDiscretization`` (2D)."""

    def __init__(
        self,
        nodes: NDArray,
        rho: float,
        degree: int,
        bounds: list[tuple[float, float]],
        sdf: Callable[[NDArray], NDArray] | None = None,
        backend: str = "numpy",
        n_edge_gauss: int = 3,
    ) -> None:
        self._nodes = np.asarray(nodes, dtype=np.float64)
        self._n_dof = int(self._nodes.shape[0])
        self._dim = int(self._nodes.shape[1])
        if self._dim != 2:
            raise NotImplementedError("MeshlessSCNIDiscretization supports 2D only (#1139).")
        self._rho = float(rho)
        self._degree = int(degree)
        self._backend = backend
        self._exps = monomial_exponents(self._dim, degree)

        cells = clipped_voronoi_cells(self._nodes, bounds, sdf)
        self._V = np.array([c.area for c in cells], dtype=np.float64)  # (N,)
        # phi at the nodes (un-smoothed MLS values), Phi[a, i] = phi_i(x_a); used by advection.
        self._phi_nodes, _ = shape_functions_and_grads(self._nodes, self._nodes, self._rho, self._exps, backend)

        # Smoothed nodal gradients B~_d, shape (N, N), via batched edge quadrature.
        self._Btilde = self._build_smoothed_gradients(cells, n_edge_gauss)

    # --- smoothed nodal gradient ---------------------------------------------
    def _build_smoothed_gradients(self, cells, n_edge_gauss):
        xi, wi = np.polynomial.legendre.leggauss(n_edge_gauss)  # [-1, 1]
        t = 0.5 * (xi + 1.0)  # edge parameter in [0, 1]
        pts, cell_idx, nw = [], [], []  # quad points, owning cell, outward-normal * edge weight
        for a, cell in enumerate(cells):
            poly = cell.polygon
            m = len(poly)
            for i in range(m):
                v0, v1 = poly[i], poly[(i + 1) % m]
                d = v1 - v0
                length = float(np.linalg.norm(d))
                if length < 1e-14:
                    continue
                normal = np.array([d[1], -d[0]]) / length  # outward for a CCW polygon
                gpts = v0[None, :] + t[:, None] * d[None, :]  # (nq, 2)
                gw = 0.5 * length * wi  # leggauss weight scaled to the edge
                pts.append(gpts)
                cell_idx.append(np.full(gpts.shape[0], a))
                nw.append(gw[:, None] * normal[None, :])  # (nq, 2)
        pts = np.vstack(pts)
        cell_idx = np.concatenate(cell_idx)
        nw = np.vstack(nw)
        phi_q, _ = shape_functions_and_grads(pts, self._nodes, self._rho, self._exps, self._backend)  # (Q, N)

        Btilde = []
        for dd in range(self._dim):
            acc = np.zeros((self._n_dof, self._n_dof))
            np.add.at(acc, cell_idx, nw[:, dd][:, None] * phi_q)  # sum_{q in cell a} nw_d phi_j
            acc /= self._V[:, None]  # divide by V_a
            Btilde.append(sparse.csr_matrix(acc))
        return Btilde

    # --- WeakFormDiscretization protocol -------------------------------------
    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dof_coordinates(self) -> NDArray:
        return self._nodes

    @property
    def rho(self) -> float:
        return self._rho

    def smoothed_gradient(self) -> list[sparse.csr_matrix]:
        """The SCNI smoothed nodal gradient operators ``[B~_0, B~_1]`` (each ``(N, N)``)."""
        return self._Btilde

    def stiffness(self) -> sparse.csr_matrix:
        Vdiag = sparse.diags(self._V)
        K = sum((B.T @ Vdiag @ B) for B in self._Btilde)
        return K.tocsr()

    def mass(self) -> sparse.csr_matrix:
        return sparse.diags(self._V).tocsr()  # lumped nodal volumes

    def gradient_projection(self) -> list[sparse.csr_matrix]:
        # R_d = diag(V) B~_d, so the solver's G_d = diag(1/M_lumped) R_d = B~_d.
        Vdiag = sparse.diags(self._V)
        return [(Vdiag @ B).tocsr() for B in self._Btilde]

    def advection(self, velocity: NDArray) -> sparse.csr_matrix:
        # C[i,j] = sum_a V_a phi_i(x_a) (alpha(x_a) . B~[a,j]); only the trial phi_j is smoothed.
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.ndim == 1:
            velocity = velocity[None, :]
        G = sparse.csr_matrix((self._n_dof, self._n_dof))
        for dd in range(self._dim):
            G = G + sparse.diags(velocity[dd]) @ self._Btilde[dd]  # (alpha_d at node a) * B~_d[a,j]
        Phi = sparse.csr_matrix(self._phi_nodes)  # Phi[a, i] = phi_i(x_a)
        return (Phi.T @ sparse.diags(self._V) @ G).tocsr()

    def boundary_shape_data(self, x_b: NDArray, normals: NDArray) -> tuple[NDArray, NDArray]:
        """MLS phi and normal-projected grad at boundary points (Nitsche is SCNI-independent)."""
        x_b = np.asarray(x_b, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        phi_b, grad_b = shape_functions_and_grads(x_b, self._nodes, self._rho, self._exps, self._backend)
        gn_b = np.einsum("qjd,qd->qj", grad_b, normals)
        return phi_b, gn_b
