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

import numbers
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    import skfem

    from numpy.typing import NDArray

    from mfgarchon.geometry.boundary import BCSegment, BoundaryConditions


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


def _segment_boundary_facets(mesh: skfem.Mesh, segment: BCSegment) -> NDArray:
    """The boundary facets a segment's DOFs come from: its named boundary, or the whole boundary for ``None``.

    One rule for the Dirichlet DOFs (`_find_segment_dofs`) and for what a ``default_bc`` is left to govern
    (`refuse_what_fem_cannot_impose`). Only the ``boundary`` name is read. A segment also placed by
    ``region``, ``region_name``, ``sdf_region`` or ``normal_direction`` resolves as its ``boundary`` face
    if it names one -- the whole face, not the placed part -- and as the whole boundary if it names none.
    """
    boundary_name = getattr(segment, "boundary", None)
    if boundary_name:
        # Issue #1489 (F3): a NAMED boundary absent from the mesh is an ERROR — silently falling back
        # to the WHOLE boundary would over-constrain (a one-wall Dirichlet applied everywhere). The
        # mesh_adapter auto-tags axis-aligned walls (x_min/x_max/...) for box domains (#607).
        if not mesh.boundaries or boundary_name not in mesh.boundaries:
            available = sorted(mesh.boundaries) if mesh.boundaries else "none"
            raise ValueError(
                f"Segment '{getattr(segment, 'name', '?')}' names boundary '{boundary_name}', "
                f"but the mesh has no such tagged boundary (available: {available}). A missing named "
                f"boundary would otherwise silently apply the BC to the ENTIRE boundary (Issue #1489). "
                f"Name one of the mesh's tagged boundaries as boundary=."
            )
        return np.asarray(mesh.boundaries[boundary_name])
    return np.asarray(mesh.boundary_facets())


def _find_segment_dofs(
    mesh: skfem.Mesh,
    basis: skfem.Basis,
    segment,
) -> list[int]:
    """DOF indices for a BCSegment on the skfem mesh, from `_segment_boundary_facets`.

    Issue #1489 (F2): resolved via ``basis.get_dofs`` on the boundary FACETS (which includes P2 edge-midpoint
    DOFs), NOT ``mesh.boundary_nodes()`` (vertices only -> half the P2 boundary left free -> Dirichlet silently
    enforced on vertices only).
    """
    return list(basis.get_dofs(_segment_boundary_facets(mesh, segment)).flatten())


def refuse_what_fem_cannot_impose(mesh: skfem.Mesh, bc: BoundaryConditions | None, consumer: str) -> None:
    """Refuse, at construction, what the FEM adapter would otherwise misread in silence (#2593).

    - **A Dirichlet value it cannot read** (`dirichlet_constant`). It used to become 0.
    - **A DIRICHLET ``default_bc`` left to govern boundary facets no segment claims.** The adapter imposes
      conditions segment by segment and never reads ``default_bc``, so those facets got the natural condition
      instead: a silent wall where an exit or a prescribed value was asked for.
    Under a DIRICHLET default, the facets each segment claims follow `_segment_boundary_facets`, the rule its
    Dirichlet DOFs follow, for segments of every type, so a segment named by anything but a tag the mesh carries -- a library alias
    such as ``"left"`` included -- raises there rather than claiming the whole boundary. A segment also
    placed by ``region``, ``region_name``, ``sdf_region`` or ``normal_direction`` claims its whole
    ``boundary`` face, or the whole boundary if it names none, which this check cannot see past.
    ``dirichlet_bc(g)`` passes, since its segment names the whole boundary.
    """
    from mfgarchon.geometry.boundary.types import BCType

    if bc is None:
        return
    for segment in bc.segments:
        if segment.bc_type == BCType.DIRICHLET and not callable(segment.value):
            dirichlet_constant(segment)
    if bc.default_bc != BCType.DIRICHLET:
        return
    claimed: set[int] = set()
    for segment in bc.segments:
        claimed.update(int(f) for f in _segment_boundary_facets(mesh, segment))
    unclaimed = sorted({int(f) for f in mesh.boundary_facets()} - claimed)
    if unclaimed:
        raise NotImplementedError(
            f"{consumer}: default_bc=DIRICHLET governs {len(unclaimed)} boundary facet(s) that no segment "
            "names, and the FEM path imposes conditions only from segments, so those facets would get the "
            "natural condition instead of the Dirichlet one (#2593). Give each Dirichlet face its own "
            "BCSegment(boundary=...)."
        )


def _evaluate_segment_values(
    segment,
    basis: skfem.Basis,
    dofs: list[int],
) -> list[float]:
    """The segment's Dirichlet value at the given DOFs, refusing a value it cannot read.

    A callable is read as ``value(x)`` at each DOF, with no time argument, and its values are not checked.
    Anything else is a constant, read by `dirichlet_constant`. This used to read every value but a plain
    ``int`` or ``float`` as 0, so ``np.float32(0.7)`` held 0 (#2593). The FDM ghost path
    (`applicator_fdm.segment_value`) reads a callable as ``value(t)`` and checks no type: the two disagree
    until #2512's row B2 gives the value one owner.
    """
    value = getattr(segment, "value", 0.0)

    if callable(value):
        # Issue #1489 (F5): evaluate at the DOF coordinates via basis.doflocs (a coordinate for EVERY
        # DOF), NOT mesh.p (vertex coordinates only). For P2 the boundary DOF set includes edge-midpoint
        # indices >= n_vertices, so mesh.p[:, dofs] was out of bounds / wrong-coordinate.
        coords = basis.doflocs[:, dofs].T  # (n_dofs, dim)
        return [float(value(x)) for x in coords]
    return [dirichlet_constant(segment)] * len(dofs)


def dirichlet_constant(segment: BCSegment) -> float:
    """A non-callable Dirichlet value as a float (#2593).

    ``None`` and anything the library's one owner of "verifiably zero" reads as zero (`_describe_bc_value`:
    an all-zero array included) is 0, as on every other path. Otherwise a finite real number or a finite 0-d
    real array is its value. A boolean, and anything else, is refused.
    """
    from mfgarchon.geometry.boundary.bc_utils import _describe_bc_value

    value = getattr(segment, "value", 0.0)
    if isinstance(value, (bool, np.bool_)) or (isinstance(value, np.ndarray) and value.dtype == np.bool_):
        raise NotImplementedError(
            f"Dirichlet segment '{getattr(segment, 'name', '?')}' has a boolean value, which is not a boundary "
            "datum: give a real number, a 0-d real array, or a callable g(x) (#2593)."
        )
    if _describe_bc_value(value) is None:
        return 0.0
    real = isinstance(value, numbers.Real) or (
        isinstance(value, np.ndarray)
        and value.ndim == 0
        and np.issubdtype(value.dtype, np.number)
        and not np.iscomplexobj(value)
    )
    if not real:
        raise NotImplementedError(
            f"Dirichlet segment '{getattr(segment, 'name', '?')}' has a value of type {type(value).__name__}, "
            "which the FEM path cannot read: give a real number, a 0-d real array, or a callable g(x) (#2593)."
        )
    constant = float(value)
    if not np.isfinite(constant):
        raise NotImplementedError(
            f"Dirichlet segment '{getattr(segment, 'name', '?')}' has a non-finite value {constant}; the FEM "
            "path would impose it and return a non-finite solution (#2593)."
        )
    return constant


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
    from mfgarchon.geometry.boundary.types import BCType

    def _is_inhomogeneous(*segments):
        """Is any of these boundary data NOT verifiably zero?

        Delegated, never re-implemented. ``describe_inhomogeneous_bc_data`` is this
        repository's single owner for that predicate, and its docstring records why: #1686's
        hole was a guard reading only ``segments``, and #1802 wrote a second copy and
        reproduced the same hole 300 lines away. A local ``isinstance(g, (int, float))``
        would reject ``np.float32(0.0)`` -- a homogeneous wall -- and accept nothing it
        should. This asks the owner instead, per segment: the ``default_bc`` channel is deliberately
        excluded here because this function assembles from segments and cannot honour it (#2305).
        """
        return describe_inhomogeneous_bc_data(
            SimpleNamespace(segments=segments, default_bc=None, default_value=None),
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
            # `"5"`, `b"5"`, `np.array("5")`, `Decimal`. Widening one guard while fixing another is
            # how a fix ships a second defect, and a first attempt here caught only the plain `str`.
            # The widening that IS wanted is numeric dtypes -- np.float32, np.int64, a 0-d float
            # array -- so the test is "is this a real number", not "can float() parse it".
            # `Fraction` is admitted deliberately: it IS a real number, and a rational boundary datum
            # is a legitimate thing to write. `Decimal` is not `numbers.Real` and stays out.
            # Only the DTYPE is checked, not the rank: `numpy>=2.0` (pyproject.toml:38) removed
            # size-1 array coercion, so `float()` already raises TypeError for every non-0-d array
            # -- measured on 1-d/1-elem, 1-d/2-elem and 2-d. A rank check here was unreachable, and
            # a mutation deleting it survived the suite, which is what unreachable looks like from
            # the outside. The dtype check is NOT redundant: `float(np.array("5"))` succeeds.
            if isinstance(g, np.ndarray) and not np.issubdtype(g.dtype, np.number):
                raise TypeError(f"boundary value is a {g.dtype} array")
            if not isinstance(g, (np.ndarray, numbers.Real)):
                raise TypeError(f"boundary value is not a real number: {type(g).__name__}")
            g = float(g)
            # A non-finite datum assembles a NaN load and every downstream solve returns NaN with no
            # error -- the failure this module's guards exist to convert into a refusal. `main`
            # already admitted `float("nan")` and `np.float64("nan")` (np.float64 subclasses float),
            # so this closes a pre-existing path as well as the 0-d-array one the widening opened.
            if not np.isfinite(g):
                raise TypeError(f"boundary value is not finite: {g}")
        except (TypeError, ValueError):
            raise NotImplementedError(
                f"{kind} segment '{segment.name}' has a non-constant value ({type(g).__name__}). "
                f"Only a constant g is implemented for the FEM {kind} boundary load; callable / "
                "BCValueProvider data is deferred (Issue #1237)."
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
                        "are deferred (Issue #1237)."
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
