"""
Translate MFGArchon BoundaryConditions to scikit-fem BC operations.

This adapter ensures FEM solvers use the same BC framework as FDM/GFDM/particle
solvers. Users specify BC via BCSegment; this module translates to skfem operations.
Which segment governs each boundary facet, and which one a node shared by differently governed facets
takes, is the owner's answer (`BoundaryConditions.segments_for_facets`, `segment_for_shared_node`; #2512
row B1, stage B1b): this module reads it and decides nothing about placement itself.

Mapping, per governing segment:
    BCType.DIRICHLET → condense() with boundary DOFs and values
    BCType.NEUMANN   → natural BC. Homogeneous (g=0): no action, whatever the value's dtype.
                       Inhomogeneous (g!=0): assemblable ONLY where the weak form's natural
                       condition constrains the gradient — the HJB side, which integrates just
                       -D*Delta u by parts. There NEUMANN(g) is ROBIN(alpha=0, beta=1, g) and the
                       load is D*int_dOmega g phi_i. The FP side integrates div(v m) on the volume
                       basis with no facet term, so ITS natural condition is the total flux J.n;
                       the same load would impose J.n = -D*g, a different condition. FP therefore
                       declares honors_inhomogeneous_neumann = False (Issue #2294); the FP's reading
                       of the shared BC refuses a nonzero g before #1686's gate (#2512, row B3)
    BCType.NO_FLUX   → same as NEUMANN (zero normal derivative)
    BCType.ROBIN     → operator augmentation (NOT condensation): a D-scaled FacetBasis boundary
                       mass + load assembled by ``assemble_robin_terms`` and folded into the
                       weak-form operator ``M/dt + D*K`` upstream (the solver's
                       ``_robin_operator_terms`` hook). Robin dofs stay free, so the
                       condensation path here treats Robin as a no-op (Issue #1237).
    BCType.PERIODIC  → NotImplementedError (needs DOF pairing across boundaries — Issue #1237,
                       still deferred)

Issue #773: BC framework integration for FEM solvers
Issue #1237: Robin BC via weak-form operator augmentation
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    import skfem

    from numpy.typing import NDArray

    from mfgarchon.geometry.boundary import BCSegment, BoundaryConditions
    from mfgarchon.geometry.boundary.conditions import BoundaryFacets


def apply_bc_to_fem_system(
    A: sparse.csr_matrix,
    rhs: NDArray,
    basis: skfem.Basis,
    bc: BoundaryConditions | None,
    homogeneous: bool = False,
    *,
    t: float | None = None,
) -> tuple[sparse.csr_matrix, NDArray]:
    """
    Apply BoundaryConditions to assembled FEM system (A, rhs).

    Condenses the Dirichlet DOFs (`get_dirichlet_dofs_and_values`). Facets governed by Neumann/no-flux take the
    natural BC in the weak form: no action. An INHOMOGENEOUS Neumann still owes a boundary load on the HJB side;
    that is an operator augmentation like Robin, assembled by ``assemble_robin_terms`` and folded in upstream, not
    applied here (Issue #2294). For Robin: no action here — the Robin boundary mass + load are an operator
    augmentation assembled by ``assemble_robin_terms`` and folded into ``M/dt + D*K`` upstream (Robin dofs stay free,
    so they are not condensed). A facet governed by Periodic raises ``NotImplementedError`` (Issue #1237, still
    deferred) — fail loud rather than silently degrade to Neumann.

    Args:
        A: System matrix (N_dof, N_dof)
        rhs: Right-hand side vector (N_dof,)
        basis: scikit-fem Basis
        bc: MFGArchon BoundaryConditions (or None for default no-flux)
        t: The time of the level solved for, at which a Dirichlet value is read (#2512 row B2). With none, a
            value that declares ``t`` is refused; a constant or ``g(x)`` needs none.

    Returns:
        (A_modified, rhs_modified) — may be condensed (smaller) or same size
    """
    if bc is None:
        # Default: no-flux (Neumann) everywhere — natural BC, no action
        return A, rhs

    _refuse_what_fem_has_no_imposition_for(_facet_groups(basis.mesh, bc))
    dof_array, val_array = get_dirichlet_dofs_and_values(basis, bc, t=t)
    if len(dof_array):
        # Issue #1489 (S2): homogeneous=True zeroes the boundary lift. The Newton CORRECTION has a
        # homogeneous boundary increment (delta[dofs]=0, since U_current already carries u=g), so
        # lifting by the actual Dirichlet values g would add a spurious -A[int,dofs]@g term and
        # corrupt every interior value. The linear solve keeps homogeneous=False (u=g is the solution).
        if homogeneous:
            val_array = np.zeros(len(dof_array))
        interior = np.setdiff1d(np.arange(A.shape[0]), dof_array)

        A_int = A[np.ix_(interior, interior)]
        rhs_int = rhs[interior] - A[np.ix_(interior, dof_array)] @ val_array

        return A_int, rhs_int

    return A, rhs


def _refuse_what_fem_has_no_imposition_for(groups: list[tuple[BCSegment, NDArray]]) -> None:
    """Refuse a facet governed by a condition this path cannot impose: PERIODIC, and any type with no arm here."""
    from mfgarchon.geometry.boundary.types import BCType

    imposable = {BCType.DIRICHLET, BCType.NEUMANN, BCType.NO_FLUX, BCType.REFLECTING, BCType.ROBIN}
    for segment, _ in groups:
        if segment.bc_type == BCType.PERIODIC:
            raise NotImplementedError(
                f"Periodic BC on segment '{segment.name}' is not implemented for the FEM solver "
                "path (needs DOF identification across paired boundaries; see Issue #1237)."
            )
        if segment.bc_type not in imposable:
            # Issue #1260: EXTRAPOLATION_LINEAR / EXTRAPOLATION_QUADRATIC (and any future BCType
            # added without a matching branch) must fail loud rather than silently degrade to
            # natural (Neumann) BC — the same design intent as Robin/Periodic above (#1241).
            raise NotImplementedError(
                f"BC type '{segment.bc_type.value}' on segment '{segment.name}' is not implemented "
                "for the FEM solver path. EXTRAPOLATION_LINEAR/QUADRATIC are ghost-cell FDM concepts "
                "with no direct FEM counterpart. Use a Dirichlet or Neumann/no-flux BC for FEM, "
                "or use an FDM/GFDM solver for extrapolation boundaries (Issue #1260)."
            )


def get_dirichlet_dofs_and_values(
    basis: skfem.Basis,
    bc: BoundaryConditions | None,
    *,
    t: float | None = None,
) -> tuple[NDArray, NDArray]:
    """
    The Dirichlet DOF indices, each once, and their values read at ``t`` (as in `apply_bc_to_fem_system`).

    A DOF is Dirichlet when the segment it takes is: the segment governing its facets, or, on a node that facets
    governed by different segments share, the owner's choice among them (`segment_for_shared_node`). So two
    Dirichlet segments meeting at a node give it one value, the chosen segment's.

    Returns:
        (dof_indices, values) — empty arrays if no Dirichlet BC.
    """
    if bc is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    dirichlet_dofs: list[NDArray] = []
    dirichlet_values: list[float] = []
    for segment, dofs in _dirichlet_segments_at_dofs(basis, bc):
        dirichlet_dofs.append(dofs)
        dirichlet_values.extend(_evaluate_segment_values(segment, basis, dofs, t))
    if not dirichlet_dofs:
        return np.array([], dtype=int), np.array([], dtype=float)
    return np.concatenate(dirichlet_dofs).astype(int), np.array(dirichlet_values, dtype=float)


def has_dirichlet_dofs(basis: skfem.Basis, bc: BoundaryConditions | None) -> bool:
    """Does the BC condense any DOF (`get_dirichlet_dofs_and_values`)? Reads no value."""
    return bc is not None and bool(_dirichlet_segments_at_dofs(basis, bc))


def is_pure_neumann(mesh: skfem.Mesh, bc: BoundaryConditions | None) -> bool:
    """Is every boundary facet governed by a natural condition (Neumann, no-flux, reflecting)? The owner's answer
    per facet, ``default_bc`` included (`BoundaryConditions.segments_for_facets`)."""
    if bc is None:
        return True

    from mfgarchon.geometry.boundary.types import BCType

    natural = {BCType.NEUMANN, BCType.NO_FLUX, BCType.REFLECTING}
    return all(segment.bc_type in natural for segment, _ in _facet_groups(mesh, bc))


#: `_boundary_facets` per mesh, by ``id``: the entry is evicted when the mesh is collected (`weakref.finalize`), so a
#: later mesh at the same address never reads it. A mesh's tags are read once, so they are not changed afterwards.
_FACETS: dict[int, tuple[NDArray, BoundaryFacets]] = {}


def _boundary_facets(mesh: skfem.Mesh) -> tuple[NDArray, BoundaryFacets]:
    """The mesh's boundary facet ids, ascending, and what the owner reads of them: midpoints, outward unit normals,
    and the rows each tag of the mesh holds."""
    cached = _FACETS.get(id(mesh))
    if cached is None:
        cached = _FACETS[id(mesh)] = _read_boundary_facets(mesh)
        weakref.finalize(mesh, _FACETS.pop, id(mesh), None)
    return cached


def _read_boundary_facets(mesh: skfem.Mesh) -> tuple[NDArray, BoundaryFacets]:
    from mfgarchon.geometry.boundary.conditions import BoundaryFacets

    ids = np.sort(np.asarray(mesh.boundary_facets(), dtype=np.int64))
    corners = mesh.p[:, mesh.facets[:, ids]]  # (dim, vertices per facet, n)
    midpoints = corners.mean(axis=1).T
    inward = mesh.p[:, mesh.t[:, mesh.f2t[0, ids]]].mean(axis=1).T - midpoints  # toward the element's centroid
    dim = mesh.p.shape[0]
    if dim == 1:
        normals = np.ones((len(ids), 1))
    elif dim == 2:
        tangent = (corners[:, 1, :] - corners[:, 0, :]).T
        normals = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    else:
        # The cross product of the two diagonals: a triangle's (vertex 2 - vertex 0 against 1 - 0) or a quad's.
        last = 3 if corners.shape[1] > 3 else 2
        normals = np.cross((corners[:, 2, :] - corners[:, 0, :]).T, (corners[:, last, :] - corners[:, 1, :]).T)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    normals[np.einsum("ij,ij->i", normals, inward) > 0] *= -1.0
    tags: dict[str, NDArray] = {}
    off_boundary = set()
    for name, members in (mesh.boundaries or {}).items():
        members = np.asarray(members, dtype=np.int64)
        on = np.isin(members, ids)
        if not on.all():
            off_boundary.add(name)
        tags[name] = np.searchsorted(ids, members[on])
    return ids, BoundaryFacets(midpoints, normals, tags, frozenset(off_boundary))


def _facet_groups(mesh: skfem.Mesh, bc: BoundaryConditions) -> list[tuple[BCSegment, NDArray]]:
    """Each segment that governs a boundary facet, with the facet ids it governs (the owner's answer)."""
    ids, facets = _boundary_facets(mesh)
    answers = bc.segments_for_facets(facets)
    groups: dict[int, tuple[BCSegment, list[int]]] = {}
    for facet, segment in zip(ids, answers, strict=True):
        groups.setdefault(id(segment), (segment, []))[1].append(int(facet))
    return [(segment, np.array(members, dtype=np.int64)) for segment, members in groups.values()]


def _dirichlet_segments_at_dofs(basis: skfem.Basis, bc: BoundaryConditions) -> list[tuple[BCSegment, NDArray]]:
    """Each DIRICHLET segment with the DOFs it holds: the DOFs of the facets it governs, less the nodes it shares
    with facets of another segment that the owner gives to that one (`segment_for_shared_node`).

    Issue #1489 (F2): DOFs come from ``basis.get_dofs`` on the boundary FACETS (which includes P2 edge-midpoint
    DOFs), NOT ``mesh.boundary_nodes()`` (vertices only -> half the P2 boundary left free -> Dirichlet silently
    enforced on vertices only).
    """
    from mfgarchon.geometry.boundary.types import BCType

    groups = _facet_groups(basis.mesh, bc)
    held = []
    for k, (_, facets) in enumerate(groups):
        group_dofs = np.asarray(basis.get_dofs(facets).flatten(), dtype=np.int64)
        held.append(np.column_stack([group_dofs, np.full(len(group_dofs), k, dtype=np.int64)]))
    pairs = np.unique(np.concatenate(held), axis=0)  # (dof, group), each pair once, sorted by dof
    dofs, first, count = np.unique(pairs[:, 0], return_index=True, return_counts=True)
    owner = pairs[first, 1].copy()
    for i in np.flatnonzero(count > 1):
        sharing = [groups[k][0] for k in pairs[first[i] : first[i] + count[i], 1]]
        where = f"boundary facets at the mesh node {np.round(basis.doflocs[:, dofs[i]], 12).tolist()}"
        chosen = bc.segment_for_shared_node(sharing, where)
        owner[i] = next(k for k, (segment, _) in enumerate(groups) if segment is chosen)
    return [
        (segment, dofs[owner == k])
        for k, (segment, _) in enumerate(groups)
        if segment.bc_type == BCType.DIRICHLET and (owner == k).any()
    ]


def refuse_what_fem_cannot_impose(basis: skfem.Basis, bc: BoundaryConditions | None, consumer: str) -> None:
    """Refuse, at construction, what the FEM adapter would otherwise misread in silence (#2593, #2595).

    - **A Dirichlet value it cannot read** (`segment_value`, #2512 row B2). It used to become 0.
    - **What the owner refuses on this mesh**: a ``boundary`` name that is no tag of the mesh and aliases no face it
      tags, a segment that governs no facet and is not effect-neutral, ``default_bc`` unset or ROBIN where a facet
      falls through to it, and a node shared by differently governed facets under a blending ``corner_strategy``
      (`BoundaryConditions.segments_for_facets`, `segment_for_shared_node`).
    - **A facet governed by a condition this path has no imposition for** (PERIODIC, EXTRAPOLATION_*).
    """
    from mfgarchon.geometry.boundary.types import BCType, segment_value

    if bc is None:
        return
    groups = _facet_groups(basis.mesh, bc)
    for segment, _ in groups:
        if segment.bc_type == BCType.DIRICHLET and not callable(segment.value):
            segment_value(segment, t=None, x=None, where=f"{consumer}, Dirichlet segment {segment.name!r}")
    _refuse_what_fem_has_no_imposition_for(groups)
    _dirichlet_segments_at_dofs(basis, bc)  # every node shared by differently governed facets


def _evaluate_segment_values(
    segment: BCSegment,
    basis: skfem.Basis,
    dofs: NDArray,
    t: float | None,
) -> list[float]:
    """The segment's Dirichlet value at the given DOFs, read by the one owner (`segment_value`, #2512 row B2).

    A callable is ``g(t, x)``, ``g(x)`` or ``g(t)``, read at each DOF and at the time of the level solved for;
    one that declares ``t`` is refused when no time is given. A constant follows the owner's rules (#2593): a real number or a real array of one element is its value, ``None``
    and what `_describe_bc_value` calls zero are 0, and anything else is refused.
    """
    from mfgarchon.geometry.boundary.types import segment_value

    where = f"the FEM Dirichlet assembly of segment {segment.name!r}"
    if callable(segment.value):
        # Issue #1489 (F5): evaluate at the DOF coordinates via basis.doflocs (a coordinate for EVERY
        # DOF), NOT mesh.p (vertex coordinates only). For P2 the boundary DOF set includes edge-midpoint
        # indices >= n_vertices, so mesh.p[:, dofs] was out of bounds / wrong-coordinate.
        coords = basis.doflocs[:, dofs].T  # (n_dofs, dim)
        return [segment_value(segment, t=t, x=x, where=where, time_keyword="t") for x in coords]
    return [segment_value(segment, t=t, x=None, where=where, time_keyword="t")] * len(dofs)


def assemble_robin_terms(
    basis: skfem.Basis,
    bc: BoundaryConditions | None,
    D: float,
    *,
    natural_bc: str = "flux",
) -> tuple[sparse.csr_matrix, NDArray] | tuple[None, None]:
    r"""Assemble the Robin operator augmentation for the weak-form diffusion operator.

    A Robin condition :math:`\alpha u + \beta\,\partial u/\partial n = g` contributes a
    boundary term when the diffusion operator :math:`-D\,\Delta u` is integrated by parts:
    :math:`-\int_{\partial\Omega} D\,(\partial u/\partial n)\, v`. Substituting
    :math:`\partial u/\partial n = (g - \alpha u)/\beta` moves a boundary MASS to the operator
    and a boundary LOAD to the RHS:

    - ``A_robin``   :math:`= D\,(\alpha/\beta)\int_{\partial\Omega}\phi_i\phi_j`  (boundary mass)
    - ``rhs_robin`` :math:`= D\,(1/\beta)\int_{\partial\Omega} g\,\phi_i`          (boundary load)

    Both scale with ``D`` exactly like the stiffness ``K`` does, so the caller adds ``A_robin``
    to ``M/dt + D*K`` and ``rhs_robin`` to each timestep RHS. The boundary mass is symmetric, so
    the FP Robin term is the adjoint (identical) of the HJB one (Type-A duality preserved).

    Summed over the facets each ``BCType.ROBIN`` segment governs (`BoundaryConditions.segments_for_facets`), each
    segment carrying its own ``alpha``, ``beta`` and constant ``value`` (``g``), **and over those of every
    inhomogeneous** ``BCType.NEUMANN`` **segment**, a NEUMANN ``default_bc`` with its ``default_value`` included
    (Issue #2294) **when the caller passes** ``natural_bc="gradient"``: for a weak form that
    integrates only ``-D*Delta u`` by parts, ``du/dn = g`` is ``ROBIN(alpha=0, beta=1, g)`` and
    contributes the load and no mass. Those coefficients are synthesized, never read off the
    segment — ``BCSegment`` defaults ``alpha=1.0, beta=0.0``, the Dirichlet weighting.
    ``natural_bc="flux"`` (the default, because it is the restrictive one) **refuses** an
    inhomogeneous Neumann: a weak form whose boundary term is the total flux ``J.n`` cannot impose
    ``dm/dn = g``, and assembling the same load there would silently impose ``J.n = -D*g``.
    ``ROBIN(alpha=0, beta=1, g)`` is still assembled under ``"flux"``, and by the FP's Robin convention
    it means exactly that, ``J.n = -D*g``: a condition on the flux, not a second spelling of ``dm/dn = g``
    (#2512, user ruling 2026-10-10).
    ``NO_FLUX`` and ``REFLECTING`` are excluded throughout: both are homogeneous here, and
    ``NO_FLUX`` on the FP side is ``J.n = 0``, owned by ``FPResolver``.

    Returns ``(None, None)`` when nothing contributes — no Robin segments and no Neumann segment
    with ``g != 0`` — so the natural/Dirichlet paths stay byte-unchanged. Callable /
    ``BCValueProvider`` ``g`` and ``beta == 0`` (pure Dirichlet) fail loud — only constant ``g``
    is implemented (Issue #1237).
    """
    if bc is None:
        return None, None

    from types import SimpleNamespace

    import skfem
    from skfem import BilinearForm, FacetBasis, LinearForm

    from mfgarchon.geometry.boundary.bc_utils import describe_inhomogeneous_bc_data
    from mfgarchon.geometry.boundary.types import BCType, segment_robin_coefficients, segment_value

    def _is_inhomogeneous(*segments):
        """Is any of these boundary data NOT verifiably zero?

        Delegated, never re-implemented. ``describe_inhomogeneous_bc_data`` is this
        repository's single owner for that predicate, and its docstring records why: #1686's
        hole was a guard reading only ``segments``, and #1802 wrote a second copy and
        reproduced the same hole 300 lines away. A local ``isinstance(g, (int, float))``
        would reject ``np.float32(0.0)`` -- a homogeneous wall -- and accept nothing it
        should. This asks the owner instead, per governing segment: a facet that falls through to ``default_bc``
        arrives as the face owner's default segment, so the default channel is asked through it.
        """
        return describe_inhomogeneous_bc_data(
            SimpleNamespace(segments=segments, default_bc=None, default_value=None),
            bc_types={BCType.NEUMANN},
        )

    boundary_segments = [
        (segment, facets)
        for segment, facets in _facet_groups(basis.mesh, bc)
        if segment.bc_type in (BCType.ROBIN, BCType.NEUMANN)
    ]
    if not boundary_segments:
        return None, None

    mesh = basis.mesh
    elem = basis.elem
    n_dof = int(basis.N)

    @BilinearForm
    def boundary_mass(u, v, w):
        return u * v

    @LinearForm
    def boundary_load(v, w):
        return v

    A_robin = sparse.csr_matrix((n_dof, n_dof))
    rhs_robin = np.zeros(n_dof)

    contributed = False
    for segment, facets in boundary_segments:
        if segment.bc_type == BCType.NEUMANN and not _is_inhomogeneous(segment):
            # Verifiably zero: contributes neither mass nor load. Skipping keeps `(None, None)`
            # for every homogeneous natural-BC solve, so those stay on the code path they were
            # on before #2294. The owner -- not `isinstance` -- decides "is this zero", which is
            # why `np.float32(0.0)`, `np.int64(0)` and an all-zero array remain the no-op they
            # have always been.
            continue

        kind = "Robin" if segment.bc_type == BCType.ROBIN else "Neumann"

        # The load is a constant g per segment (#1237): read by the one owner, with no time or point to pass.
        where = f"the FEM {kind} load of segment {segment.name!r}, one constant per segment (#1237),"
        g = segment_value(segment, t=None, x=None, where=where)

        if segment.bc_type == BCType.NEUMANN:
            # Issue #2294: `du/dn = g` is `ROBIN(alpha=0, beta=1, g)` -- the same mathematical
            # condition, so it owes the same boundary load `D * int_dOmega g phi_i` and no boundary
            # mass. The arm in `apply_bc_to_fem_system` is a bare `pass` because a natural BC is not
            # a condensation; that is correct and is not where the term was missing.
            #
            # The coefficients are SYNTHESIZED, never read off the segment: `BCSegment` defaults
            # alpha=1.0, beta=0.0 (the Dirichlet weighting), so a Neumann segment's own alpha/beta
            # describe a different condition entirely and using them would divide by zero.
            if natural_bc != "gradient":
                raise NotImplementedError(
                    f"Inhomogeneous Neumann segment '{segment.name}' (g={g}) on a weak form whose "
                    "natural boundary condition is the TOTAL FLUX J.n, not the gradient. Adding the "
                    "load D*int g phi here would impose J.n = -D*g, which equals dm/dn = g only "
                    "where the drift has no normal component at the wall -- a different condition "
                    "wearing the same name (Issues #2294, #1237)."
                )
            alpha, beta = 0.0, 1.0
        else:
            # The coefficients are read by the one owner (#2512 row B2): a constant each, by the constant contract; a
            # callable refused naming #1237, and a provider with no iteration state to resolve it refused (#1979).
            alpha, beta = segment_robin_coefficients(
                segment, where=f"the FEM Robin operator of segment {segment.name!r}"
            )
            if beta == 0.0:
                raise NotImplementedError(
                    f"Robin segment '{segment.name}' has beta=0, i.e. a pure Dirichlet condition "
                    "(alpha*u = g). Use BCType.DIRICHLET for that; a Robin term requires beta != 0 "
                    "(Issue #1237)."
                )

        contributed = True
        fb = FacetBasis(mesh, elem, facets=facets)

        M_bnd = skfem.asm(boundary_mass, fb)
        load_bnd = skfem.asm(boundary_load, fb)

        A_robin = A_robin + D * (alpha / beta) * M_bnd
        rhs_robin = rhs_robin + D * (g / beta) * load_bnd

    # A bc holding only homogeneous Neumann segments reaches here having contributed nothing. It
    # must return the same `(None, None)` as a bc with no segments at all, or the caller adds an
    # all-zero matrix and vector and the solve stops being byte-identical to the one before #2294.
    if not contributed:
        return None, None

    return A_robin.tocsr(), rhs_robin
