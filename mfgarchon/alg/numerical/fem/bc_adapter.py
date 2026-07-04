"""
Translate MFGArchon BoundaryConditions to scikit-fem BC operations.

This adapter ensures FEM solvers use the same BC framework as FDM/GFDM/particle
solvers. Users specify BC via BCSegment; this module translates to skfem operations.

Mapping:
    BCType.DIRICHLET → condense() with boundary DOFs and values
    BCType.NEUMANN   → natural BC (default in weak form, no action needed)
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
    For Neumann/no-flux: no action (natural BC in weak form).
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
            values = _evaluate_segment_values(segment, mesh, dofs)
            dirichlet_dofs.extend(dofs)
            dirichlet_values.extend(values)

        elif segment.bc_type in (BCType.NEUMANN, BCType.NO_FLUX, BCType.REFLECTING):
            # Natural BC — no action needed in weak form
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
            values = _evaluate_segment_values(segment, mesh, dofs)
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

    # mesh.boundaries is None when no named regions were tagged (mesh_adapter tags axis-aligned
    # walls x_min/x_max/... for box domains; #607). Guard it so an untagged mesh falls back
    # cleanly instead of raising "argument of type 'NoneType' is not iterable".
    if boundary_name and mesh.boundaries and boundary_name in mesh.boundaries:
        # Named boundary region
        facets = mesh.boundaries[boundary_name]
        dofs = basis.get_dofs(facets)
        return list(dofs.flatten())

    # Fallback: use all boundary nodes
    return list(mesh.boundary_nodes())


def _evaluate_segment_values(
    segment,
    mesh: skfem.Mesh,
    dofs: list[int],
) -> list[float]:
    """Evaluate BCSegment value at the given DOFs."""
    value = getattr(segment, "value", 0.0)

    if callable(value):
        # Value is a function: evaluate at DOF coordinates
        coords = mesh.p[:, dofs].T  # (n_dofs, dim)
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

    Summed over all ``BCType.ROBIN`` segments; each carries ``alpha``, ``beta``, and a constant
    ``value`` (``g``). Returns ``(None, None)`` when there are no Robin segments (no-op, so the
    natural/Dirichlet paths are byte-unchanged). Callable / ``BCValueProvider`` ``g`` and
    ``beta == 0`` (pure Dirichlet) fail loud — only constant ``g`` is implemented (Issue #1237).
    """
    if bc is None:
        return None, None

    import skfem
    from skfem import BilinearForm, FacetBasis, LinearForm

    from mfgarchon.geometry.boundary.types import BCType

    robin_segments = [s for s in bc.segments if s.bc_type == BCType.ROBIN]
    if not robin_segments:
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

    for segment in robin_segments:
        alpha = float(getattr(segment, "alpha", 1.0))
        beta = float(getattr(segment, "beta", 0.0))
        if beta == 0.0:
            raise NotImplementedError(
                f"Robin segment '{segment.name}' has beta=0, i.e. a pure Dirichlet condition "
                "(alpha*u = g). Use BCType.DIRICHLET for that; a Robin term requires beta != 0 "
                "(Issue #1237)."
            )

        g = getattr(segment, "value", 0.0)
        if not isinstance(g, (int, float)):
            raise NotImplementedError(
                f"Robin segment '{segment.name}' has a non-constant value ({type(g).__name__}). "
                "Only a constant g is implemented for the FEM Robin boundary load; callable / "
                "BCValueProvider Robin data is deferred (Issue #1237). For an adjoint-consistent "
                "(state-dependent) Robin BC, resolve the provider to a constant before the solve."
            )
        g = float(g)

        facets = _find_segment_facets(mesh, segment)
        fb = FacetBasis(mesh, elem, facets=facets)

        M_bnd = skfem.asm(boundary_mass, fb)
        load_bnd = skfem.asm(boundary_load, fb)

        A_robin = A_robin + D * (alpha / beta) * M_bnd
        rhs_robin = rhs_robin + D * (g / beta) * load_bnd

    return A_robin.tocsr(), rhs_robin
