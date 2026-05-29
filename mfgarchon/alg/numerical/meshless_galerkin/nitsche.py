"""
Symmetric Nitsche Dirichlet boundary terms for the meshless-Galerkin weak form.

MLS shape functions are non-interpolatory (``u_h(x_i) = sum_j phi_j(x_i) U_j != U_i``),
so Dirichlet data cannot be imposed by nodal condensation. This module assembles the
symmetric (SIPG-type) Nitsche boundary operators that impose ``u = g`` (HJB) /
``m = 0`` (FP, absorbing) weakly, added to the diffusion block of the implicit operator.

For the diffusion operator ``-D*Laplace`` with Dirichlet data ``g`` on ``Gamma_D``,
integration by parts gives the symmetric Nitsche boundary block (added to ``+D*K``)::

    N = -D*B - D*B^T + (gamma*D/rho) * P

with, at surface quadrature points ``{x_b, w_b, n_b}`` on ``Gamma_D``,

- ``P[i,j] = sum_b w_b phi_i(x_b) phi_j(x_b)``                (boundary Gram, symmetric)
- ``B[i,j] = sum_b w_b phi_i(x_b) (n_b . grad phi_j(x_b))``   (normal flux, non-symmetric)

and the HJB Dirichlet-data load (moved to the RHS; uses the EXACT ``g(x_b)``, not the
MLS reconstruction ``g_h(x_b)``, which would degrade consistency)::

    f[i] = -D sum_b w_b (n_b . grad phi_i(x_b)) g(x_b)
           + (gamma*D/rho) sum_b w_b phi_i(x_b) g(x_b)

The penalty length scale is the MLS support radius ``rho`` (not the node pitch ``h``):
the shape functions vary on scale ``rho``. The dimensionless coercivity condition is
``gamma > 2*C_tr`` (``C_tr`` the local discrete trace-inverse constant); ``gamma = 20``
is the degree-2 default. ``N`` is symmetric (``N = N^T``), so the HJB and FP solvers
carry the IDENTICAL block -- this is what makes the Type-A transpose identity
``A_FP = A_HJB^T`` hold on the diffusion + Nitsche sub-block. FP absorbing (``m = 0``)
is the ``g = 0`` case, so it adds no load.

Scope (interim, #1138):

- ``Gamma_D`` = flagged faces of the cloud's axis-aligned bounding box (axis-aligned
  outward normals). Full ``B(x_i, rho) intersect Omega`` clipping is #1139.
- ``disc.advection`` carries no boundary term, so ``Gamma_D`` is **diffusively
  absorbing but advectively reflecting** -- rigorous for ``b.n = 0`` on ``Gamma_D``.
  Advective outflow (e.g. evacuation) needs an upwind boundary flux: a follow-up.
- Inhomogeneous FP-Dirichlet (``m = g != 0``) is out of scope (the FP solve loop has
  no Nitsche-RHS hook; only the homogeneous absorbing ``g = 0`` case is supported).

Issue #1138.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from mfgarchon.alg.numerical.meshless_galerkin.quadrature import boundary_tensor_gauss, surface_quadrature

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.alg.numerical.meshless_galerkin.discretization import MeshlessGalerkinDiscretization
    from mfgarchon.geometry.boundary import BoundaryConditions


def dirichlet_segments(bc: BoundaryConditions | None) -> list:
    """Dirichlet segments of a BoundaryConditions object (empty list if none)."""
    if bc is None:
        return []
    from mfgarchon.geometry.boundary.types import BCType

    return [s for s in bc.segments if s.bc_type == BCType.DIRICHLET]


def _domain_bounds(disc: MeshlessGalerkinDiscretization) -> list[tuple[float, float]]:
    X = disc.dof_coordinates
    return [(float(X[:, k].min()), float(X[:, k].max())) for k in range(X.shape[1])]


def _segment_faces(segment, d: int) -> list[tuple[int, str]]:
    """Bounding-box faces a Dirichlet segment applies to.

    A named face (``"x_min"``) maps to one ``(axis, side)``; an unscoped segment
    (``boundary is None``) applies to every face of the box. Anything else (e.g. a
    Gmsh physical-group tag) is unsupported in the interim bounding-box path.
    """
    from mfgarchon.geometry.boundary.types import parse_boundary_face

    face = parse_boundary_face(segment.boundary)
    if face is not None:
        return [(face.axis, face.side)]
    if segment.boundary is None:
        return [(ax, side) for ax in range(d) for side in ("min", "max")]
    raise NotImplementedError(
        f"Dirichlet boundary {segment.boundary!r} is not an axis-aligned bounding-box face; "
        "only named faces (e.g. 'x_min') or unscoped (all faces) are supported by the interim "
        "meshless Nitsche path (#1138)."
    )


def _check_boundary_node_coverage(x_b: NDArray, disc) -> None:
    """Fail fast if any boundary quadrature point lacks MLS node support.

    A point with fewer than ``len(exps)`` cloud nodes within the support radius makes
    the MLS moment matrix singular; raise a greppable error naming the bare point
    rather than letting a deep ``LinAlgError`` surface later. The cloud must cover the
    Dirichlet boundary ``{sdf=0}``.
    """
    from scipy.spatial import cKDTree

    n_min = len(disc._exps)
    counts = np.asarray(cKDTree(disc.dof_coordinates).query_ball_point(x_b, disc.rho, return_length=True))
    bad = np.flatnonzero(counts < n_min)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"Curved-boundary quadrature point {np.round(x_b[i], 4).tolist()} has only {int(counts[i])} cloud "
            f"nodes within rho={disc.rho:.4g} (need >= {n_min}); the cloud must cover the Dirichlet boundary "
            "{sdf=0}. Add nodes near the boundary or enlarge delta (#1139)."
        )


def _segment_quadrature(segment, disc, bounds, n_gauss):
    """Boundary quadrature ``(x_b, w_b, n_b)`` for one Dirichlet segment.

    A segment carrying ``sdf_region`` (a curved boundary, #1139) is integrated on the
    level set ``{sdf_region = 0}`` via ``surface_quadrature``; the background resolution
    mirrors the cloud scale (``n_cells ~ max bbox side / rho``). Otherwise the existing
    axis-aligned bounding-box face rule (``boundary_tensor_gauss``) is used.
    """
    sdf = getattr(segment, "sdf_region", None)
    if sdf is not None:
        # Boundary marching grid finer than the support radius for a smooth boundary
        # curve; the boundary points still need rho-coverage (checked below).
        max_side = max(b - a for a, b in bounds)
        n_cells = max(16, int(np.ceil(2.0 * max_side / disc.rho)))
        x_b, w_b, n_b = surface_quadrature(sdf, bounds, n_cells)
        _check_boundary_node_coverage(x_b, disc)
        return x_b, w_b, n_b
    faces = _segment_faces(segment, disc.dim)
    return boundary_tensor_gauss(bounds, faces, n_gauss=n_gauss)


def _evaluate_g(value, x_b: NDArray) -> NDArray:
    """Prescribed Dirichlet value at boundary points (scalar or callable g(x))."""
    if isinstance(value, (int, float)):
        return np.full(x_b.shape[0], float(value))
    if callable(value):
        return np.array([float(value(x)) for x in x_b], dtype=np.float64)
    raise NotImplementedError(
        f"Dirichlet value of type {type(value).__name__} is unsupported in the meshless Nitsche "
        "path; use a float or a callable g(x) (#1138)."
    )


def assemble_nitsche_terms(
    disc: MeshlessGalerkinDiscretization,
    bc: BoundaryConditions | None,
    D: float,
    gamma: float,
    n_gauss: int,
    include_data: bool,
) -> tuple[sparse.csr_matrix | None, NDArray | None]:
    """Symmetric Nitsche operator block (and HJB Dirichlet-data RHS) for the weak form.

    Args:
        disc: the meshless discretization (provides ``rho`` and ``boundary_shape_data``).
        bc: boundary conditions; Dirichlet segments select ``Gamma_D``.
        D: diffusion coefficient (``sigma^2 / 2``); every Nitsche term scales with it.
        gamma: dimensionless penalty (coercivity needs ``gamma > 2*C_tr``).
        n_gauss: Gauss points per free dimension for the surface quadrature.
        include_data: ``True`` (HJB ``u=g``) builds the data RHS; ``False`` (FP
            absorbing ``g=0``) returns ``rhs=None``.

    Returns:
        ``(N, rhs)``: ``N`` the ``(n_dof, n_dof)`` sparse block to ADD to
        ``M/dt + D*K``; ``rhs`` the ``(n_dof,)`` Dirichlet-data load or ``None``.
        ``(None, None)`` if there are no Dirichlet segments.
    """
    segs = dirichlet_segments(bc)
    if not segs:
        return None, None

    bounds = _domain_bounds(disc)
    rho = disc.rho

    xs, ws, ns, gs = [], [], [], []
    for s in segs:
        x_b, w_b, n_b = _segment_quadrature(s, disc, bounds, n_gauss)
        xs.append(x_b)
        ws.append(w_b)
        ns.append(n_b)
        gs.append(_evaluate_g(s.value, x_b) if include_data else np.zeros(x_b.shape[0]))

    x_b = np.vstack(xs)
    w_b = np.concatenate(ws)
    n_b = np.vstack(ns)
    g_b = np.concatenate(gs)

    phi_b, gn_b = disc.boundary_shape_data(x_b, n_b)  # (Q_b, N) each

    # Sparse assembly: MLS shape functions are compactly supported, so only dofs whose
    # support reaches Gamma_D are nonzero; keep the boundary block sparse.
    phi_sp = sparse.csr_matrix(phi_b)
    gn_sp = sparse.csr_matrix(gn_b)
    W = sparse.diags(w_b)
    P = phi_sp.T @ W @ phi_sp  # P[i,j] = sum_b w_b phi_i phi_j  (symmetric)
    B = phi_sp.T @ W @ gn_sp  # B[i,j] = sum_b w_b phi_i (n.grad phi_j)
    pen = gamma * D / rho
    N = ((-D) * B + (-D) * B.T + pen * P).tocsr()

    rhs = None
    if include_data and np.any(g_b != 0.0):
        wg = w_b * g_b
        rhs = (-D) * (gn_b.T @ wg) + pen * (phi_b.T @ wg)

    return N, rhs
