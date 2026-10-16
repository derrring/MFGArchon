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
- Both solvers assemble the data load from the BC they hold. The FP holds the shared BC as
  read through ``bc_utils.fp_view_of_shared_bc``, where a Dirichlet is an exit (``g = 0``,
  absorbing; #2512, convention row 5), or its own BC, where ``MeshlessGalerkinFPSolver._validate_bc_support``
  refuses a Dirichlet value (#2532). Either way the FP's data load is zero: ``m = 0``.
- Which face each condition governs is the face owner's answer, ``BoundaryConditions.segment_for_face``
  (#2512 row B1, stage B1c-2): a face label or a ``normal_direction`` governs its face, and a face no segment
  governs takes ``default_bc`` (#2490). A ``region``, ``sdf_region`` or marked-region ``region_name`` placement
  that decides a Dirichlet condition is refused: the owner decides it on the problem's grid, not on these faces
  (#2717); a condition on part of a face is #2712.

Issue #1138.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from mfgarchon.alg.numerical.meshless_galerkin.quadrature import boundary_tensor_gauss

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.alg.numerical.meshless_galerkin.discretization import MeshlessGalerkinDiscretization
    from mfgarchon.geometry.boundary import BoundaryConditions
    from mfgarchon.geometry.boundary.types import BCSegment


def refuse_what_nitsche_cannot_place(bc: BoundaryConditions | None, d: int) -> None:
    """Refuse, at construction, a BC whose placement the face owner refuses -- the terms are assembled at the
    first solve, and this asks at construction what they would ask (`_dirichlet_faces`).

    The owner (`BoundaryConditions.segment_for_face`, #2512 row B1) refuses a segment that covers part of a face
    ahead of the one governing the rest with another condition, one that governs no face, a placement it cannot
    decide without a grid, and a face nothing governs when ``default_bc`` cannot (unset, or ROBIN).
    """
    if bc is None:
        return
    _dirichlet_faces(bc, d)


def places_no_dirichlet_face(bc: BoundaryConditions | None, d: int) -> bool:
    """No face carries a Dirichlet condition once precedence is applied: the natural condition everywhere.

    The meshless pair's pure-Neumann answer, read from the placement Nitsche imposes with, so the two cannot
    disagree. A Dirichlet segment whose every face a higher-priority segment claims is not a wall; a
    type-only reading (FEM's `is_pure_neumann` until #2595) called it one, and with no Nitsche term to impose
    the solve fell through to nodal condensation, which this basis cannot do.
    """
    return not any(faces for _, faces in _dirichlet_faces(bc, d))


def _domain_bounds(disc: MeshlessGalerkinDiscretization) -> list[tuple[float, float]]:
    X = disc.dof_coordinates
    return [(float(X[:, k].min()), float(X[:, k].max())) for k in range(X.shape[1])]


def _dirichlet_faces(bc: BoundaryConditions | None, d: int) -> list[tuple[BCSegment, list[tuple[int, str]]]]:
    """Each Dirichlet condition with the bounding-box faces it governs, in the BC's priority order: the face
    owner's answer (`BoundaryConditions.segment_for_face`, #2512 row B1, stage B1c-2).

    The owner applies the precedence, places face labels and ``normal_direction``, refuses what it cannot place,
    and gives a face no segment governs to ``default_bc``. A DIRICHLET default is therefore imposed, with
    ``default_value``, on the faces it governs. The segments the owner would place on a grid's nodes rather than
    on these faces (`BoundaryConditions.segments_placed_off_face_labels`) are refused before it is asked.
    """
    from mfgarchon.geometry.boundary.types import BCType, BoundaryFace

    if bc is None:
        return []
    if bc.dimension is None:
        # The owner refuses a segment that governs no face only on a BC that knows its dimension, so a BC built
        # without one is asked as the discretisation's (#2717, review 1).
        bc = bc.bind_dimension(d)
    off_labels = bc.segments_placed_off_face_labels()
    if off_labels:
        names = ", ".join(repr(seg.name) for seg in off_labels)
        raise NotImplementedError(
            f"Meshless Galerkin: segment(s) {names} are placed by region, sdf_region or a marked region_name, where "
            "their placement decides a Dirichlet condition. The owner decides such a placement on a grid's nodes, not "
            "on the box faces Nitsche integrates, so it is refused as before #2512's stage B1c-2 (#2717). Place the "
            "condition with a face label -- boundary='x_max', region_name='x_max', or 'all' -- or normal_direction "
            "on an axis-aligned face; a condition on part of a face is #2712."
        )
    by_rank: dict[int, tuple[BCSegment, list[tuple[int, str]]]] = {}
    for axis in range(d):
        for side in ("min", "max"):
            segment = bc.segment_for_face(BoundaryFace(axis, side))
            if segment.bc_type != BCType.DIRICHLET:
                continue
            # The default is built afresh on each query, so it is keyed after every segment, not by identity.
            rank = next((i for i, s in enumerate(bc.segments) if s is segment), len(bc.segments))
            by_rank.setdefault(rank, (segment, []))[1].append((axis, side))
    return [by_rank[rank] for rank in sorted(by_rank)]


def _segment_quadrature(faces, disc, bounds, n_gauss):
    """Boundary quadrature ``(x_b, w_b, n_b)`` on the faces one Dirichlet segment governs.

    The axis-aligned bounding-box face rule (``boundary_tensor_gauss``), with the cell
    count sized off the cloud scale: **two cells per support radius**, so at least ``2 * n_gauss`` points fall within every ``rho`` along a face.
    The integrand is ``phi_i phi_j`` and ``phi_i (n . grad phi_j)``, which varies on the
    MLS support scale ``rho`` and therefore gets *finer* under refinement; a rule whose
    resolution does not follow it resolves less of the integrand at every level (#1679).

    The factor of two is where the accuracy plateaus, not a copied constant. On the
    #1679 manufactured Poisson at n=21 the solution error reads 6.7375e-05 at one cell
    per radius and 5.9721e-05 at two, while four and eight both give 5.9726e-05.

    Deliberately conservative for an anisotropic box: ``max_side`` is the largest side
    of the bounding box, while ``boundary_tensor_gauss`` applies ``n_cells`` to each
    face's own free axes. It therefore never under-resolves, but a box of 1 x 0.01
    over-refines its short faces. Making that per-face needs a per-face ``n_cells``,
    which this function cannot express today.
    """
    max_side = max(b - a for a, b in bounds)
    # Cells per support radius, not per domain: rho shrinks with the cloud, so this
    # count grows under refinement while a fixed one silently does not.
    n_cells = max(1, int(np.ceil(2.0 * max_side / disc.rho)))
    return boundary_tensor_gauss(bounds, faces, n_cells=n_cells, n_gauss=n_gauss)


def _evaluate_g(segment, x_b: NDArray, t: float | None) -> NDArray:
    """A segment's prescribed Dirichlet value at boundary points and at ``t``, read by the one owner
    (`segment_value`, #2512 row B2): a constant, or a callable ``g(t, x)``, ``g(x)`` or ``g(t)``. One that declares
    ``t`` is refused when no time is given."""
    from mfgarchon.geometry.boundary.types import segment_value

    where = f"the meshless Nitsche assembly of segment {segment.name!r}"
    if callable(segment.value):
        return np.array(
            [segment_value(segment, t=t, x=x, where=where, time_keyword="t") for x in x_b], dtype=np.float64
        )
    return np.full(x_b.shape[0], segment_value(segment, t=t, x=None, where=where, time_keyword="t"), dtype=np.float64)


def assemble_nitsche_terms(
    disc: MeshlessGalerkinDiscretization,
    bc: BoundaryConditions | None,
    D: float,
    gamma: float,
    n_gauss: int,
    t: float | None = None,
) -> tuple[sparse.csr_matrix | None, NDArray | None]:
    """Symmetric Nitsche operator block and Dirichlet-data RHS for the weak form, the data read at ``t``.

    Args:
        disc: the meshless discretization (provides ``rho`` and ``boundary_shape_data``).
        bc: boundary conditions; Dirichlet segments select ``Gamma_D``.
        D: diffusion coefficient (``sigma^2 / 2``); every Nitsche term scales with it.
        gamma: dimensionless penalty (coercivity needs ``gamma > 2*C_tr``).
        n_gauss: Gauss points per free dimension for the surface quadrature.

    Returns:
        ``(N, rhs)``: ``N`` the ``(n_dof, n_dof)`` sparse block to ADD to
        ``M/dt + D*K``; ``rhs`` the ``(n_dof,)`` Dirichlet-data load, or ``None`` when every value is
        zero -- the FP's case, whose shared Dirichlet the translator has set to the absorbing ``g = 0``.
        ``(None, None)`` if no Dirichlet face is placed (`_dirichlet_faces`).
    """
    placed = [(seg, faces) for seg, faces in _dirichlet_faces(bc, disc.dim) if faces]
    if not placed:
        return None, None

    bounds = _domain_bounds(disc)
    rho = disc.rho

    xs, ws, ns, gs = [], [], [], []
    for s, faces in placed:
        x_b, w_b, n_b = _segment_quadrature(faces, disc, bounds, n_gauss)
        xs.append(x_b)
        ws.append(w_b)
        ns.append(n_b)
        gs.append(_evaluate_g(s, x_b, t))

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
    if np.any(g_b != 0.0):
        wg = w_b * g_b
        rhs = (-D) * (gn_b.T @ wg) + pen * (phi_b.T @ wg)

    return N, rhs
