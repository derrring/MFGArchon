"""
Meshless Galerkin (MLS) implementation of the weak-form discretization protocol.

``MeshlessGalerkinDiscretization`` assembles the weak-form operators on a
scattered point cloud by local quadrature against MLS shape functions, with no
mesh. It is the meshfree counterpart of ``FEMDiscretization``; both satisfy
``WeakFormDiscretization`` so a weak-form solver consumes either.

Shape functions and gradients are evaluated once at the quadrature points (for
stiffness/mass/advection) and once at the nodes (for the gradient projection);
operators are then contractions over the cached arrays. Dimension enters only
through the contracted index.

Issue #1131 Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from mfgarchon.alg.numerical.meshless_galerkin.mls_basis import (
    monomial_exponents,
    shape_functions_and_grads,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MeshlessGalerkinDiscretization:
    """Point cloud + MLS implementation of ``WeakFormDiscretization``.

    Args:
        nodes: collocation/test centers, shape ``(n_dof, dim)``.
        rho: MLS support radius.
        degree: MLS polynomial degree ``k``.
        quad_points: integration points, shape ``(Q, dim)``.
        quad_weights: integration weights, shape ``(Q,)``.
        backend: ``"numpy"`` (analytic, default) or ``"jax"`` (autodiff).
    """

    def __init__(
        self,
        nodes: NDArray,
        rho: float,
        degree: int,
        quad_points: NDArray,
        quad_weights: NDArray,
        backend: str = "numpy",
    ) -> None:
        self._nodes = np.asarray(nodes, dtype=np.float64)
        self._n_dof = int(self._nodes.shape[0])
        self._dim = int(self._nodes.shape[1])
        self._rho = float(rho)
        self._backend = backend
        self._w = np.asarray(quad_weights, dtype=np.float64)
        self._degree = int(degree)
        self._exps = monomial_exponents(self._dim, degree)

        # (phi, grad) at quadrature points: (Q, N), (Q, N, dim)
        self._phi, self._grad = shape_functions_and_grads(
            np.asarray(quad_points, dtype=np.float64), self._nodes, self._rho, self._exps, backend
        )

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
        """MLS support radius (the length scale for the Nitsche penalty)."""
        return self._rho

    def boundary_shape_data(self, x_b: NDArray, normals: NDArray) -> tuple[NDArray, NDArray]:
        """MLS shape functions and normal-projected gradients at boundary points.

        Evaluates ``phi`` and ``grad(phi)`` at the surface quadrature points
        ``x_b`` (shape ``(Q_b, dim)``) and contracts the gradient with the
        per-point outward unit normals (shape ``(Q_b, dim)``). Returns
        ``(phi_b, gn_b)`` of shape ``(Q_b, n_dof)`` with
        ``phi_b[b, i] = phi_i(x_b)`` and ``gn_b[b, j] = n_b . grad phi_j(x_b)``.
        These are the primitives for the Nitsche boundary operators ``B`` and
        ``P`` (see ``meshless_galerkin/nitsche.py``).
        """
        x_b = np.asarray(x_b, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        phi_b, grad_b = shape_functions_and_grads(x_b, self._nodes, self._rho, self._exps, self._backend)
        gn_b = np.einsum("qjd,qd->qj", grad_b, normals)
        return phi_b, gn_b

    def stiffness(self) -> sparse.csr_matrix:
        K = np.einsum("q,qid,qjd->ij", self._w, self._grad, self._grad)
        return sparse.csr_matrix(K)

    def mass(self) -> sparse.csr_matrix:
        M = np.einsum("q,qi,qj->ij", self._w, self._phi, self._phi)
        return sparse.csr_matrix(M)

    def advection(self, velocity: NDArray) -> sparse.csr_matrix:
        # velocity at dofs (dim, n_dof) or (n_dof,); interpolate to quad points
        # via the MLS shape functions: alpha(xi_q) = sum_j phi_j(xi_q) v_j.
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.ndim == 1:
            velocity = velocity[None, :]
        alpha_q = np.einsum("qj,dj->qd", self._phi, velocity)  # (Q, dim)
        a = np.einsum("qd,qjd->qj", alpha_q, self._grad)  # (Q, N): alpha . grad phi_j
        C = np.einsum("q,qi,qj->ij", self._w, self._phi, a)
        return sparse.csr_matrix(C)

    def gradient_projection(self) -> list[sparse.csr_matrix]:
        # Weak-form derivative R_d[i, j] = int phi_i (d phi_j / d x_d) dx (the
        # WeakFormDiscretization protocol contract): the solver recovers the nodal
        # gradient via the mass-lumped projection G_d = M_lumped^{-1} R_d. Returning
        # the strong pointwise derivative d phi_j / d x_d(x_i) here makes that
        # M_lumped^{-1} a spurious second factor (~1/dx blow-up). Issue #1145.
        return [
            sparse.csr_matrix(np.einsum("q,qi,qj->ij", self._w, self._phi, self._grad[:, :, d]))
            for d in range(self._dim)
        ]


def discretization_from_cloud(
    collocation_points: NDArray,
    delta: float,
    degree: int = 2,
    n_gauss: int = 4,
    backend: str = "numpy",
    domain: object | None = None,
) -> MeshlessGalerkinDiscretization:
    """Build a discretization from a point cloud with interior tensor-Gauss quadrature.

    The background-grid resolution is ~ one cell per node per dimension (assumes a
    quasi-uniform cloud). ``delta`` is the MLS support radius; ``backend`` selects
    the numpy (default) or jax derivative engine.

    ``domain`` is an optional point predicate ``(N, d) -> bool`` (``True`` = inside
    Omega), composable with ``mfgarchon.geometry.predicates`` (``sphere_region``,
    ``sdf_region``, ...). When given, the background tensor-Gauss is **masked to
    Omega**, so the Galerkin operators integrate over a non-rectangular (Lipschitz)
    domain rather than the cloud bounding box. Default ``None`` keeps the full
    bounding box (rectangular Omega).

    Masking is *required* for well-posedness on a non-rectangular Omega: unclipped
    quadrature points fall outside the node cloud where the MLS moment matrix is
    singular. The cloud must therefore **cover** Omega; if near-boundary points lack
    node support the MLS solve raises ``LinAlgError`` (fail-fast, no silent fallback).
    The structural identities (``K = K^T``, mass conservation, ``A_FP = A_HJB^T``) are
    algebraic and hold for any quadrature, so masking preserves them; only boundary
    accuracy is affected (crude mask is ~O(h^1.5); high-order moment-fitting is #1139).
    """
    from mfgarchon.alg.numerical.meshless_galerkin.quadrature import tensor_gauss

    nodes = np.asarray(collocation_points, dtype=np.float64)
    if nodes.ndim == 1:
        nodes = nodes[:, None]
    n, d = nodes.shape
    n_per = max(2, round(n ** (1.0 / d)))
    bounds = [(float(nodes[:, k].min()), float(nodes[:, k].max())) for k in range(d)]
    pts, wts = tensor_gauss(bounds, n_cells=n_per - 1, n_gauss=n_gauss)
    if domain is not None:
        mask = np.asarray(domain(pts), dtype=bool)
        if not mask.any():
            raise ValueError("domain predicate excluded every quadrature point; check the cloud and domain.")
        pts, wts = pts[mask], wts[mask]
    return MeshlessGalerkinDiscretization(nodes, delta, degree, pts, wts, backend=backend)


if __name__ == "__main__":
    """Smoke test: numpy/jax backends agree; protocol invariants hold (1D, 2D)."""
    from scipy.sparse import linalg as sla

    from mfgarchon.alg.numerical.meshless_galerkin.quadrature import tensor_gauss
    from mfgarchon.alg.numerical.weak_form_discretization import WeakFormDiscretization

    def grid(d, n):
        ax = np.linspace(0.0, 1.0, n)
        mesh = np.meshgrid(*([ax] * d), indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)

    for d in (1, 2):
        n_per = 11 if d == 1 else 7
        nodes = grid(d, n_per)
        h = 1.0 / (n_per - 1)
        rho = 3.5 * h if d == 1 else 2.6 * h
        pts, wts = tensor_gauss([(0.0, 1.0)] * d, n_cells=n_per - 1, n_gauss=4)

        disc = MeshlessGalerkinDiscretization(nodes, rho, 2, pts, wts, backend="numpy")
        assert isinstance(disc, WeakFormDiscretization), "protocol conformance failed"

        K = disc.stiffness().toarray()
        one = np.ones(disc.n_dof)
        K_sym = np.linalg.norm(K - K.T) / np.linalg.norm(K)
        K_one = np.max(np.abs(K @ one))
        # gradient projection is the weak-form R_d (Issue #1145); the solver recovers
        # the nodal gradient via G_d = M_lumped^{-1} R_d, which on a linear field
        # u = x_e reproduces delta_{ec}.
        M_lumped_inv = 1.0 / disc.mass().toarray().sum(axis=1)
        G = [M_lumped_inv[:, None] * r.toarray() for r in disc.gradient_projection()]
        grad_err = max(
            float(np.max(np.abs(G[e] @ nodes[:, c] - (1.0 if e == c else 0.0)))) for e in range(d) for c in range(d)
        )
        alpha = -np.asarray(pts).T  # velocity at dofs is (dim, N); use nodes-consistent shape
        v = -nodes.T  # (dim, N): drift -x at dofs (potential MFG, Psi=|x|^2/2)
        A_one = np.max(np.abs((0.05 * disc.stiffness() + disc.advection(v)) @ one))

        # numpy vs jax agreement (skip cleanly if jax absent).
        try:
            disc_j = MeshlessGalerkinDiscretization(nodes, rho, 2, pts, wts, backend="jax")
            backend_diff = sla.norm(disc.stiffness() - disc_j.stiffness()) / np.linalg.norm(K)
            jax_note = f"{backend_diff:.2e}"
        except ImportError:
            jax_note = "jax not installed (skipped)"

        print(f"d={d}: N={disc.n_dof}")
        print(f"   ||K-K^T||/||K|| = {K_sym:.2e}   ||K@1|| = {K_one:.2e}")
        print(f"   grad-projection err = {grad_err:.2e}   ||(nuK+C)@1|| = {A_one:.2e}")
        print(f"   numpy-vs-jax stiffness rel diff = {jax_note}")
    print("MeshlessGalerkinDiscretization smoke test complete.")
