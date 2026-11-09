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
from mfgarchon.types.callable_protocols import read_region

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

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
        # Issue #1485: this is the Gauss-quadrature ASSEMBLY path -> guard the MLS moment-matrix
        # conditioning (SCNI's own discretization does not, by design -- see mls_basis.py).
        self._phi, self._grad = shape_functions_and_grads(
            np.asarray(quad_points, dtype=np.float64),
            self._nodes,
            self._rho,
            self._exps,
            backend,
            check_conditioning=True,
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

    @property
    def lumps_mass(self) -> bool:
        return False

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

    def streamline_diffusion(self, velocity: NDArray, diffusion: float, c_scale: float = 1.0) -> sparse.csr_matrix:
        r"""Symmetric SUPG streamline-diffusion operator for an advection ``velocity``.

        ``S[i,j] = sum_q w_q tau_q (b.grad phi_i)(b.grad phi_j)`` with the SUPG parameter
        ``tau = c_scale * (delta/2|b|)(coth Pe - 1/Pe)``, ``Pe = |b| delta/(2 diffusion)``,
        length scale ``delta`` = the MLS support radius. ``b`` (``(dim, n_dof)`` or
        ``(n_dof,)``) is interpolated to the quadrature points through the shape functions,
        exactly as ``advection`` interpolates its velocity.

        ``S`` is **symmetric** (an outer product of ``b.grad phi_i`` with itself) and
        ``S @ 1 = 0`` (partition of unity ``sum_j grad phi_j = 0``). Added IDENTICALLY to
        the FP advection block and the HJB Newton Jacobian, the former preserves the
        Type-A duality ``A_FP = A_HJB^T`` and the latter keeps the FP mass-conserving.
        ``diffusion`` MUST be the same ``D = sigma^2/2`` used on the other operator, or
        the duality breaks. ``c_scale == 0`` returns the zero matrix. Issue #1145 (Bug B).
        """
        if c_scale == 0.0:
            return sparse.csr_matrix((self._n_dof, self._n_dof))
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.ndim == 1:
            velocity = velocity[None, :]
        delta = self._rho  # single global MLS support radius (a scalar)
        b_q = np.einsum("qj,dj->qd", self._phi, velocity)  # (Q, dim): velocity at quad points
        b_norm = np.sqrt((b_q**2).sum(axis=1))  # (Q,)
        pe = b_norm * delta / (2.0 * diffusion)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = 1.0 / np.tanh(pe) - 1.0 / pe  # coth(Pe) - 1/Pe -> 0 as Pe -> 0
            tau = np.where(b_norm > 1e-12, c_scale * (delta / (2.0 * b_norm)) * xi, 0.0)
        b_grad = np.einsum("qd,qid->qi", b_q, self._grad)  # (Q, N): b . grad phi_i
        return sparse.csr_matrix(np.einsum("q,q,qi,qj->ij", self._w, tau, b_grad, b_grad))

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
    domain: Callable[..., Any] | None = None,
) -> MeshlessGalerkinDiscretization:
    """Build a discretization from a point cloud with interior tensor-Gauss quadrature.

    The background-grid resolution is ~ one cell per node per dimension (assumes a
    quasi-uniform cloud). ``delta`` is the MLS support radius; ``backend`` selects
    the numpy (default) or jax derivative engine.

    ``domain`` is an optional point predicate reading ``x``, a coordinate tuple (``True`` = inside
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
        mask = np.asarray(read_region(domain, pts, role="domain"), dtype=bool)
        if not mask.any():
            raise ValueError("domain predicate excluded every quadrature point; check the cloud and domain.")
        pts, wts = pts[mask], wts[mask]
    return MeshlessGalerkinDiscretization(nodes, delta, degree, pts, wts, backend=backend)
