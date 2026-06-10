"""
Translate MFGArchon BoundaryConditions to scikit-fem BC operations.

This adapter ensures FEM solvers use the same BC framework as FDM/GFDM/particle
solvers. Users specify BC via BCSegment; this module translates to skfem operations.

Mapping:
    BCType.DIRICHLET → condense() with boundary DOFs and values
    BCType.NEUMANN   → natural BC (default in weak form, no action needed)
    BCType.NO_FLUX   → same as NEUMANN (zero normal derivative)
    BCType.ROBIN     → NotImplementedError (needs a D-scaled FacetBasis boundary term threaded
                       through weak-form assembly; not resolvable in this coefficient-blind
                       adapter — Issue #1237)
    BCType.PERIODIC  → NotImplementedError (needs DOF pairing across boundaries — Issue #1237)

Issue #773: BC framework integration for FEM solvers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import skfem

    from numpy.typing import NDArray
    from scipy import sparse

    from mfgarchon.geometry.boundary import BoundaryConditions


def apply_bc_to_fem_system(
    A: sparse.csr_matrix,
    rhs: NDArray,
    basis: skfem.Basis,
    bc: BoundaryConditions | None,
) -> tuple[sparse.csr_matrix, NDArray]:
    """
    Apply BoundaryConditions to assembled FEM system (A, rhs).

    For Dirichlet segments: condense the system (eliminate boundary DOFs).
    For Neumann/no-flux: no action (natural BC in weak form).
    For Robin/Periodic: raises ``NotImplementedError`` (Issue #1237) — fail loud rather than
    silently degrade to Neumann.

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
            # Robin BC (alpha*u + beta*du/dn = g) needs a FacetBasis boundary term scaled by the
            # diffusion coefficient D, which is assembled upstream (weak-form base class) and is
            # NOT visible to this coefficient-blind adapter. Fail loud rather than silently
            # degrade to Neumann (which would run the wrong physics quietly). See Issue #1237.
            raise NotImplementedError(
                f"Robin BC on segment '{segment.name}' is not implemented for the FEM solver path. "
                "Correct Robin BC requires threading the diffusion coefficient / time factor into "
                "the weak-form assembly (Issue #1237). Use Dirichlet or Neumann/no-flux here, or an "
                "FDM/GFDM solver for Robin/reflecting boundaries."
            )

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
        val_array = np.array(dirichlet_values, dtype=float)
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
