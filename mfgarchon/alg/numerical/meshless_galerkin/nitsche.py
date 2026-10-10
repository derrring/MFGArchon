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
- A segment placed by ``region``, ``sdf_region`` or ``normal_direction`` is refused when its placement
  matters: a Dirichlet one, one that outranks a Dirichlet segment, or any with a DIRICHLET ``default_bc``.
  This path places conditions by face name only, with the BC's precedence (#2490). Otherwise such a
  segment carries the natural condition, which needs no placement.

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


def refuse_what_nitsche_cannot_place(bc: BoundaryConditions | None) -> None:
    """Refuse, at construction, a BC the meshless pair's Nitsche path would misplace.

    The terms are assembled at the first solve; this checks at construction what they would read.

    - **A Dirichlet segment this path cannot place** (``region``, ``sdf_region``, ``normal_direction``, or
      a ``region_name`` that is not a face), or any segment it cannot place that outranks a Dirichlet one,
      is refused by `_segment_faces` rather than dropped (#2490).
    - **A DIRICHLET ``default_bc`` that governs some face** is refused. Nitsche reads only segments
      (`_dirichlet_faces`, `places_no_dirichlet_face`), so before this the faces it governed got the
      natural condition: a default-DIRICHLET right wall kept all its mass (ratio 1.0000) where the same
      wall as an explicit segment absorbs (0.8326 at 11 points, delta 0.35, sigma 0.3, T 0.5, zero drift)
      (#2512 S4). Coverage is `_segment_faces` too, so a segment it cannot place is refused here rather
      than counted as covering a face.
    """
    if bc is None:
        return
    from mfgarchon.geometry.boundary.types import BCType

    _dirichlet_faces(bc, bc.dimension or 0)
    if getattr(bc, "default_bc", None) == BCType.DIRICHLET and _faces_left_to_the_default(bc):
        raise NotImplementedError(
            "Meshless Galerkin: a DIRICHLET default_bc is not imposed -- Nitsche reads only segments, so "
            "the faces it governs would get the natural condition instead. Give each Dirichlet face its "
            "own BCSegment(boundary=...)."
        )


def _faces_left_to_the_default(bc: BoundaryConditions) -> list[str]:
    """Bounding-box faces no segment governs, so ``default_bc`` would. Coverage is `_segment_faces`."""
    from mfgarchon.geometry.boundary.types import BoundaryFace

    d = bc.dimension or 0
    covered = {face for seg in bc.segments for face in _segment_faces(seg, d)}
    return [
        BoundaryFace(ax, side).to_string() for ax in range(d) for side in ("min", "max") if (ax, side) not in covered
    ]


def places_no_dirichlet_face(bc: BoundaryConditions | None, d: int) -> bool:
    """No face carries a Dirichlet condition once precedence is applied: the natural condition everywhere.

    The meshless pair's pure-Neumann answer, read from the placement Nitsche imposes with, so the two cannot
    disagree. A Dirichlet segment whose every face a higher-priority segment claims is not a wall; a
    type-only reading (`fem.bc_adapter.is_pure_neumann`) called it one, and with no Nitsche term to impose
    the solve fell through to nodal condensation, which this basis cannot do.
    """
    return not any(faces for _, faces in _dirichlet_faces(bc, d))


def _domain_bounds(disc: MeshlessGalerkinDiscretization) -> list[tuple[float, float]]:
    X = disc.dof_coordinates
    return [(float(X[:, k].min()), float(X[:, k].max())) for k in range(X.shape[1])]


def _segment_faces(segment, d: int) -> list[tuple[int, str]]:
    """Bounding-box faces a segment names, whole faces only. Which of them it governs is
    `_dirichlet_faces`'s answer, which applies the BC's precedence.

    A face named by ``boundary`` (aliases such as ``"left"`` included) or by a ``region_name`` that is a
    face label maps to that face; ``boundary=None`` or ``"all"`` names every face. A segment carrying
    ``region``, ``sdf_region`` or ``normal_direction`` is refused: this path places conditions by face name
    and cannot read those fields. It used to drop them, sending a segment with no ``boundary`` to every face
    and one with a ``boundary`` to that whole face (#2490).
    Measured before: a 1-D Dirichlet segment on ``region={0: (0.5, 1)}`` absorbed 0.6652, the both-walls
    figure, where x_max alone gives 0.8326. A Dirichlet ``sdf_region`` was integrated on its zero level
    set -- for a ball centred on a corner of the unit square, a quarter circle inside the domain.

    `bc_utils.refuse_unplaceable_segments` refuses a related set for the readers that take one condition
    per face (the semi-Lagrangian pair, FP-FDM's periodicity), with a different policy: it also refuses
    ``"all"`` and ``region_name`` segments, which this path can place.
    """
    from mfgarchon.geometry.boundary.types import BCType, parse_boundary_face

    if segment.bc_type == BCType.DIRICHLET and getattr(segment, "sdf_region", None) is not None:
        raise NotImplementedError(
            f"Meshless Galerkin: Dirichlet segment {segment.name!r} carries sdf_region, which BCSegment defines as "
            "selecting part of the boundary; this path integrated its zero level set instead -- in general not "
            "the part it selects, and for a ball on a box corner a curve inside the domain. The curved-domain "
            "Dirichlet route (#1139) is withdrawn until curved "
            "boundaries have their own field, distinct from sdf_region (#2490). On a box, name the face with "
            "boundary='x_min' etc."
        )
    restricted = [f for f in ("region", "sdf_region", "normal_direction") if getattr(segment, f, None) is not None]
    if restricted:
        raise NotImplementedError(
            f"Meshless Galerkin: segment {segment.name!r} is placed by {', '.join(restricted)}, which the interim "
            "Nitsche path cannot read -- it places conditions by face name only -- so it is refused rather than "
            "dropped (#2490). Name the face with boundary='x_min' etc."
        )
    if segment.region_name is not None:
        face = parse_boundary_face(segment.region_name)
        if face is None:
            raise NotImplementedError(
                f"Meshless Galerkin: segment {segment.name!r} names region {segment.region_name!r}, which is not a "
                "bounding-box face; the interim Nitsche path cannot place it (#2490)."
            )
        return [(face.axis, face.side)]
    if segment.boundary is None or segment.boundary == "all":
        return [(ax, side) for ax in range(d) for side in ("min", "max")]
    face = parse_boundary_face(segment.boundary)
    if face is not None:
        return [(face.axis, face.side)]
    raise NotImplementedError(
        f"Meshless Galerkin: boundary {segment.boundary!r} is not an axis-aligned bounding-box face; only named "
        "faces (e.g. 'x_min') or every face (boundary=None or 'all') are supported by the interim Nitsche path "
        "(#1138)."
    )


def _dirichlet_faces(bc: BoundaryConditions | None, d: int) -> list[tuple[BCSegment, list[tuple[int, str]]]]:
    """Each Dirichlet segment with the faces it governs: those it names that no earlier segment in the BC's
    priority order already claims -- the order `BoundaryConditions.get_bc_at_point` reads, highest first.

    A face named by both a priority -1 Dirichlet ``boundary="all"`` and a priority 1 no-flux ``x_min`` is
    the no-flux segment's; imposing the Dirichlet there too absorbed 0.6652 where the resolver's answer gives
    0.8326 (#2490). Segments after the last Dirichlet one cannot take a face from it and are not read, so a
    lower-priority segment this path cannot place is not refused here.
    """
    from mfgarchon.geometry.boundary.types import BCType

    if bc is None:
        return []
    segments = list(bc.segments)  # sorted by priority, highest first (BoundaryConditions.__post_init__)
    dirichlet_at = [i for i, seg in enumerate(segments) if seg.bc_type == BCType.DIRICHLET]
    if not dirichlet_at:
        return []
    claimed: set[tuple[int, str]] = set()
    placed = []
    for seg in segments[: dirichlet_at[-1] + 1]:
        if seg.bc_type == BCType.DIRICHLET:
            faces = _segment_faces(seg, d)
        else:
            try:
                faces = _segment_faces(seg, d)
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"Meshless Galerkin: segment {seg.name!r} outranks a Dirichlet segment in the BC's priority order, "
                    "so the faces it claims decide where the Dirichlet one applies, and this path cannot place it "
                    f"(#2490): {exc}"
                ) from exc
        if seg.bc_type == BCType.DIRICHLET:
            placed.append((seg, [face for face in faces if face not in claimed]))
        claimed.update(faces)
    return placed


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


def _evaluate_g(value, x_b: NDArray) -> NDArray:
    """Prescribed Dirichlet value at boundary points: a real number, a callable ``g(x)``, or anything the
    one owner of "verifiably zero" (`bc_utils._describe_bc_value`) counts as zero -- ``None``,
    ``np.int64(0)``, ``np.zeros(1)``. The FP reads the translated BC, which leaves such a value as it is,
    so a narrower test here would refuse a zero the translator already accepted."""
    import numbers

    from mfgarchon.geometry.boundary.bc_utils import _describe_bc_value

    if np.iscomplexobj(value):
        # `float()` on a NumPy complex drops the imaginary part with only a ComplexWarning, and
        # `_describe_bc_value` reads 1j as zero that way: refused, not truncated.
        raise NotImplementedError(
            f"Dirichlet value {value!r} is complex; the meshless Nitsche path takes a real g (#2490)."
        )
    if _describe_bc_value(value) is None:
        return np.zeros(x_b.shape[0])
    if isinstance(value, numbers.Real) or (isinstance(value, np.ndarray) and value.size == 1):
        return np.full(x_b.shape[0], float(np.asarray(value).reshape(-1)[0]))
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
) -> tuple[sparse.csr_matrix | None, NDArray | None]:
    """Symmetric Nitsche operator block and Dirichlet-data RHS for the weak form.

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
        gs.append(_evaluate_g(s.value, x_b))

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
