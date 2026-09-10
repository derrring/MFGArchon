"""
Translate MFGArchon BoundaryConditions to scikit-fem BC operations.

This adapter ensures FEM solvers use the same BC framework as FDM/GFDM/particle
solvers. Users specify BC via BCSegment; this module translates to skfem operations.

Mapping:
    BCType.DIRICHLET → condense() with boundary DOFs and values
    BCType.NEUMANN   → natural BC. Homogeneous (g=0): no action, whatever the value's dtype.
                       Inhomogeneous (g!=0): assemblable ONLY where the weak form's natural
                       condition constrains the gradient — the HJB side, which integrates just
                       -D*Delta u by parts. There NEUMANN(g) is ROBIN(alpha=0, beta=1, g) and the
                       load is D*int_dOmega g phi_i. The FP side integrates div(v m) on the volume
                       basis with no facet term, so ITS natural condition is the total flux J.n;
                       the same load would impose J.n = -D*g, a different condition. FP therefore
                       declares honors_inhomogeneous_neumann = False and #1686's gate refuses it
                       (Issue #2294)
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

import numbers
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    import skfem

    from numpy.typing import NDArray

    from mfgarchon.geometry.boundary import BoundaryConditions


def apply_bc_to_fem_system(
    A: sparse.csr_matrix,
    rhs: NDArray,
    basis: skfem.Basis,
    bc: BoundaryConditions | None,
    homogeneous: bool = False,
) -> tuple[sparse.csr_matrix, NDArray]:
    """
    Apply BoundaryConditions to assembled FEM system (A, rhs).

    For Dirichlet segments: condense the system (eliminate boundary DOFs).
    For Neumann/no-flux: no condensation (natural BC in weak form). An INHOMOGENEOUS Neumann
    still owes a boundary load on the HJB side; that is an operator augmentation like Robin,
    assembled by ``assemble_robin_terms`` and folded in upstream, not applied here (Issue #2294).
    For Robin: no action here — the Robin boundary mass + load are an operator augmentation
    assembled by ``assemble_robin_terms`` and folded into ``M/dt + D*K`` upstream (Robin dofs
    stay free, so they are not condensed). For Periodic: raises ``NotImplementedError``
    (Issue #1237, still deferred) — fail loud rather than silently degrade to Neumann.

    Args:
        A: System matrix (N_dof, N_dof)
        rhs: Right-hand side vector (N_dof,)
        basis: scikit-fem Basis
        bc: MFGArchon BoundaryConditions (or None for default no-flux)

    Returns:
        (A_modified, rhs_modified) — may be condensed (smaller) or same size
    """
    if bc is None:
        # Default: no-flux (Neumann) everywhere — natural BC, no action
        return A, rhs

    from mfgarchon.geometry.boundary.types import BCType

    dirichlet_dofs = []
    dirichlet_values = []
    mesh = basis.mesh

    for segment in bc.segments:
        if segment.bc_type in (BCType.DIRICHLET,):
            # Find DOFs on this boundary segment
            dofs = _find_segment_dofs(mesh, basis, segment)
            values = _evaluate_segment_values(segment, basis, dofs)
            dirichlet_dofs.extend(dofs)
            dirichlet_values.extend(values)

        elif segment.bc_type in (BCType.NEUMANN, BCType.NO_FLUX, BCType.REFLECTING):
            # Natural BC — nothing to CONDENSE, which is all this function does. An
            # inhomogeneous Neumann (g != 0) does owe a boundary load; it is assembled by
            # ``assemble_robin_terms`` and folded into M/dt + D*K upstream (Issue #2294).
            pass

        elif segment.bc_type == BCType.ROBIN:
            # Robin BC (alpha*u + beta*du/dn = g) is an OPERATOR AUGMENTATION, not condensation:
            # the D-scaled boundary mass (D*alpha/beta)*int_dOmega phi_i phi_j and load
            # (D/beta)*int_dOmega g phi_i are assembled by ``assemble_robin_terms`` and folded into
            # ``M/dt + D*K`` upstream (the solver's ``_robin_operator_terms`` hook). Robin dofs stay
            # free, so there is nothing to condense here. See Issue #1237.
            pass

        elif segment.bc_type == BCType.PERIODIC:
            raise NotImplementedError(
                f"Periodic BC on segment '{segment.name}' is not implemented for the FEM solver "
                "path (needs DOF identification across paired boundaries; see Issue #1237)."
            )

        else:
            # Issue #1260: EXTRAPOLATION_LINEAR / EXTRAPOLATION_QUADRATIC (and any future BCType
            # added without a matching branch) must fail loud rather than silently degrade to
            # natural (Neumann) BC — the same design intent as Robin/Periodic above (#1241).
            # 2026-06-10 audit.
            raise NotImplementedError(
                f"BC type '{segment.bc_type.value}' on segment '{segment.name}' is not implemented "
                "for the FEM solver path. EXTRAPOLATION_LINEAR/QUADRATIC are ghost-cell FDM concepts "
                "with no direct FEM counterpart. Use a Dirichlet or Neumann/no-flux BC for FEM, "
                "or use an FDM/GFDM solver for extrapolation boundaries (Issue #1260)."
            )

    if dirichlet_dofs:
        # Condense: eliminate Dirichlet DOFs from system
        dof_array = np.array(dirichlet_dofs, dtype=int)
        # Issue #1489 (S2): homogeneous=True zeroes the boundary lift. The Newton CORRECTION has a
        # homogeneous boundary increment (delta[dofs]=0, since U_current already carries u=g), so
        # lifting by the actual Dirichlet values g would add a spurious -A[int,dofs]@g term and
        # corrupt every interior value. The linear solve keeps homogeneous=False (u=g is the solution).
        val_array = np.zeros(len(dirichlet_dofs)) if homogeneous else np.array(dirichlet_values, dtype=float)
        # Issue #1489 (F4): a corner/edge DOF shared by two Dirichlet segments appears twice in the
        # list; dedup so the condensation lift `A[int,dofs]@val` does not double-count its column (and
        # the scatter-back is not order-dependent). Fail loud on conflicting values for the same DOF.
        uniq, first_idx = np.unique(dof_array, return_index=True)
        if len(uniq) != len(dof_array):
            if not homogeneous:
                for d in uniq:
                    vd = val_array[dof_array == d]
                    if not np.allclose(vd, vd[0]):
                        raise ValueError(
                            f"Conflicting Dirichlet values on shared DOF {int(d)}: {sorted(set(vd.tolist()))} "
                            f"— two BC segments meet at this corner/edge with different values (Issue #1489)."
                        )
            dof_array = uniq
            val_array = val_array[first_idx]
        interior = np.setdiff1d(np.arange(A.shape[0]), dof_array)

        A_int = A[np.ix_(interior, interior)]
        rhs_int = rhs[interior] - A[np.ix_(interior, dof_array)] @ val_array

        return A_int, rhs_int

    return A, rhs


def get_dirichlet_dofs_and_values(
    basis: skfem.Basis,
    bc: BoundaryConditions | None,
) -> tuple[NDArray, NDArray]:
    """
    Extract Dirichlet DOF indices and values from BoundaryConditions.

    Returns:
        (dof_indices, values) — empty arrays if no Dirichlet BC.
    """
    if bc is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    from mfgarchon.geometry.boundary.types import BCType

    mesh = basis.mesh
    dirichlet_dofs = []
    dirichlet_values = []

    for segment in bc.segments:
        if segment.bc_type == BCType.DIRICHLET:
            dofs = _find_segment_dofs(mesh, basis, segment)
            values = _evaluate_segment_values(segment, basis, dofs)
            dirichlet_dofs.extend(dofs)
            dirichlet_values.extend(values)

    return np.array(dirichlet_dofs, dtype=int), np.array(dirichlet_values, dtype=float)


def is_pure_neumann(bc: BoundaryConditions | None) -> bool:
    """Check if all BC segments are Neumann/no-flux (natural BC)."""
    if bc is None:
        return True

    from mfgarchon.geometry.boundary.types import BCType

    neumann_types = {BCType.NEUMANN, BCType.NO_FLUX, BCType.REFLECTING}
    return all(s.bc_type in neumann_types for s in bc.segments)


def _find_segment_dofs(
    mesh: skfem.Mesh,
    basis: skfem.Basis,
    segment,
) -> list[int]:
    """Find DOF indices for a BCSegment on the skfem mesh.

    Uses segment.boundary name to look up mesh.boundaries dict,
    or falls back to all boundary nodes.
    """
    boundary_name = getattr(segment, "boundary", None)

    if boundary_name:
        # Issue #1489 (F3): a NAMED boundary absent from the mesh is an ERROR — silently falling back
        # to the WHOLE boundary would over-constrain (a one-wall Dirichlet applied everywhere). The
        # mesh_adapter auto-tags axis-aligned walls (x_min/x_max/...) for box domains (#607).
        if not mesh.boundaries or boundary_name not in mesh.boundaries:
            available = sorted(mesh.boundaries) if mesh.boundaries else "none"
            raise ValueError(
                f"Dirichlet segment '{getattr(segment, 'name', '?')}' names boundary '{boundary_name}', "
                f"but the mesh has no such tagged boundary (available: {available}). A missing named "
                f"boundary would otherwise silently apply the BC to the ENTIRE boundary (Issue #1489). "
                f"Tag it, or use boundary=None for the whole boundary."
            )
        return list(basis.get_dofs(mesh.boundaries[boundary_name]).flatten())

    # Issue #1489 (F2): boundary=None -> the whole boundary. Resolve DOFs via basis.get_dofs on the
    # boundary FACETS (which includes P2 edge-midpoint DOFs), NOT mesh.boundary_nodes() (vertices only
    # -> half the P2 boundary is left free -> Dirichlet silently enforced on vertices only).
    return list(basis.get_dofs(mesh.boundary_facets()).flatten())


def _evaluate_segment_values(
    segment,
    basis: skfem.Basis,
    dofs: list[int],
) -> list[float]:
    """Evaluate BCSegment value at the given DOFs."""
    value = getattr(segment, "value", 0.0)

    if callable(value):
        # Issue #1489 (F5): evaluate at the DOF coordinates via basis.doflocs (a coordinate for EVERY
        # DOF), NOT mesh.p (vertex coordinates only). For P2 the boundary DOF set includes edge-midpoint
        # indices >= n_vertices, so mesh.p[:, dofs] was out of bounds / wrong-coordinate.
        coords = basis.doflocs[:, dofs].T  # (n_dofs, dim)
        return [float(value(x)) for x in coords]
    elif isinstance(value, (int, float)):
        return [float(value)] * len(dofs)
    else:
        return [0.0] * len(dofs)


def _find_segment_facets(mesh: skfem.Mesh, segment) -> NDArray:
    """Return the boundary-facet indices for a BCSegment on the skfem mesh.

    Mirrors :func:`_find_segment_dofs` but returns FACET indices (for ``FacetBasis``),
    not DOF indices: a Robin term integrates over facets, not nodes. Uses the
    ``segment.boundary`` name to look up ``mesh.boundaries`` (axis-aligned walls are tagged
    by ``meshdata_to_skfem``; #607), falling back to all boundary facets when the mesh is
    untagged or the name is unknown.
    """
    boundary_name = getattr(segment, "boundary", None)

    if boundary_name and mesh.boundaries and boundary_name in mesh.boundaries:
        return np.asarray(mesh.boundaries[boundary_name], dtype=np.int64)

    # Fallback: integrate over the whole boundary.
    return np.asarray(mesh.boundary_facets(), dtype=np.int64)


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

    Summed over all ``BCType.ROBIN`` segments, each carrying its own ``alpha``, ``beta`` and
    constant ``value`` (``g``), **and over every inhomogeneous** ``BCType.NEUMANN`` **segment**
    (Issue #2294) **when the caller passes** ``natural_bc="gradient"``: for a weak form that
    integrates only ``-D*Delta u`` by parts, ``du/dn = g`` is ``ROBIN(alpha=0, beta=1, g)`` and
    contributes the load and no mass. Those coefficients are synthesized, never read off the
    segment — ``BCSegment`` defaults ``alpha=1.0, beta=0.0``, the Dirichlet weighting.
    ``natural_bc="flux"`` (the default, because it is the restrictive one) **refuses** an
    inhomogeneous Neumann: a weak form whose boundary term is the total flux ``J.n`` cannot impose
    ``dm/dn = g``, and assembling the same load there would silently impose ``J.n = -D*g``. Note the
    scope: that refusal is keyed on the ``NEUMANN`` **spelling**. ``ROBIN(alpha=0, beta=1, g)`` is
    the same condition and is still assembled under ``"flux"``, which is pre-existing behaviour with
    its own tests and its own #1237 disclosure -- not something this parameter closes.
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
    from mfgarchon.geometry.boundary.types import BCType

    def _is_inhomogeneous(*segments, default_bc=None, default_value=None):
        """Is any of these boundary data NOT verifiably zero?

        Delegated, never re-implemented. ``describe_inhomogeneous_bc_data`` is this
        repository's single owner for that predicate, and its docstring records why: #1686's
        hole was a guard reading only ``segments``, and #1802 wrote a second copy and
        reproduced the same hole 300 lines away. A local ``isinstance(g, (int, float))``
        would reject ``np.float32(0.0)`` -- a homogeneous wall -- and accept nothing it
        should. This asks the owner instead, one channel at a time.
        """
        return describe_inhomogeneous_bc_data(
            SimpleNamespace(segments=segments, default_bc=default_bc, default_value=default_value),
            bc_types={BCType.NEUMANN},
        )

    # NOTE: this function assembles from `bc.segments` only. An inhomogeneous Neumann arriving
    # solely through the `default_bc` / `default_value` fall-through therefore reaches no assembly
    # and is dropped -- pre-existing, since `apply_bc_to_fem_system` has always been segments-only,
    # and filed rather than guarded here. Two guards were tried and both were wrong: one read the
    # default channel alone and rejected `neumann_bc(g)`, whose factory sets BOTH channels in
    # lockstep; the replacement asked whether the segments COVER the boundary, via a raw
    # `boundary in mesh.boundaries` lookup that `parse_boundary_face` was written to replace, so
    # `boundary="left"` -- the spelling this library's own BCSegment docstring uses -- resolved to
    # the whole boundary and the guard under-refused. See the tracking issue.
    boundary_segments = [s for s in bc.segments if s.bc_type in (BCType.ROBIN, BCType.NEUMANN)]
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
    for segment in boundary_segments:
        if segment.bc_type == BCType.NEUMANN and not _is_inhomogeneous(segment):
            # Verifiably zero: contributes neither mass nor load. Skipping keeps `(None, None)`
            # for every homogeneous natural-BC solve, so those stay on the code path they were
            # on before #2294. The owner -- not `isinstance` -- decides "is this zero", which is
            # why `np.float32(0.0)`, `np.int64(0)` and an all-zero array remain the no-op they
            # have always been.
            continue

        kind = "Robin" if segment.bc_type == BCType.ROBIN else "Neumann"

        g = getattr(segment, "value", 0.0)
        try:
            # `float()` alone is WIDER than the `isinstance(g, (int, float))` it replaces: it takes
            # `"5"`, `b"5"`, `np.array("5")`, `Decimal`, `Fraction`. Widening one guard while fixing
            # another is how a fix ships a second defect, and a first attempt here caught only the
            # plain `str`. The widening that IS wanted is numeric dtypes -- np.float32, np.int64, a
            # 0-d float array -- so the test is "is this a real number", not "can float() parse it".
            if isinstance(g, np.ndarray):
                if g.ndim != 0 or not np.issubdtype(g.dtype, np.number):
                    raise TypeError(f"boundary value is a {g.ndim}-d {g.dtype} array")
            elif not isinstance(g, numbers.Real):
                raise TypeError(f"boundary value is not a real number: {type(g).__name__}")
            g = float(g)
        except (TypeError, ValueError):
            raise NotImplementedError(
                f"{kind} segment '{segment.name}' has a non-constant value ({type(g).__name__}). "
                f"Only a constant g is implemented for the FEM {kind} boundary load; callable / "
                "BCValueProvider data is deferred (Issue #1237). For an adjoint-consistent "
                "(state-dependent) BC, resolve the provider to a constant before the solve."
            ) from None

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
            # Issue #1979: guard alpha and beta the way `g` is guarded above. Without this, a
            # provider-valued coefficient reaches float() and raises a bare builtin TypeError --
            # unnamed, ungreppable, and silent about what the caller should do instead.
            for _name, _coeff in (("alpha", getattr(segment, "alpha", 1.0)), ("beta", getattr(segment, "beta", 0.0))):
                if not isinstance(_coeff, (int, float)):
                    raise NotImplementedError(
                        f"Robin segment '{segment.name}' has a non-constant {_name} "
                        f"({type(_coeff).__name__}). Only a constant {_name} is implemented for the "
                        "FEM Robin operator augmentation; callable / BCValueProvider coefficients "
                        "are deferred (Issue #1237). For an adjoint-consistent (state-dependent) "
                        "Robin BC, resolve the provider to a constant before the solve -- "
                        "`bc.with_resolved_providers(state)`."
                    )
            alpha = float(getattr(segment, "alpha", 1.0))
            beta = float(getattr(segment, "beta", 0.0))
            if beta == 0.0:
                raise NotImplementedError(
                    f"Robin segment '{segment.name}' has beta=0, i.e. a pure Dirichlet condition "
                    "(alpha*u = g). Use BCType.DIRICHLET for that; a Robin term requires beta != 0 "
                    "(Issue #1237)."
                )

        contributed = True
        facets = _find_segment_facets(mesh, segment)
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
