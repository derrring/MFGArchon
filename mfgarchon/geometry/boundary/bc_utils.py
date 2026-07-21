"""
Centralized boundary condition utilities for all solver types.

Issue #702: Shared BC type detection and operation mapping for FDM, SL, GFDM, etc.

This module provides utilities that replace duplicated BC handling logic in:
- fp_fdm_time_stepping._get_bc_type()
- fp_semi_lagrangian_adjoint._get_bc_operation_type()
- hjb_semi_lagrangian._get_bc_type_string()

All solvers should import from this module for consistent BC handling.
"""

from __future__ import annotations

from typing import Any


def get_bc_type_string(boundary_conditions: Any) -> str | None:
    """
    Extract BC type string from any BoundaryConditions object.

    Supports:
    - Unified BoundaryConditions (conditions.py) with .type property
    - Legacy BoundaryConditions1DFDM with .type attribute
    - Mixed BC (returns first segment's type)

    Args:
        boundary_conditions: Any BC object

    Returns:
        BC type string (e.g., "periodic", "dirichlet", "no_flux") or None

    Example:
        >>> from mfgarchon.geometry.boundary import no_flux_bc
        >>> bc = no_flux_bc(dimension=1)
        >>> get_bc_type_string(bc)
        'no_flux'
    """
    if boundary_conditions is None:
        return None

    # Try unified BC .type property
    try:
        bc_type = boundary_conditions.type
        if bc_type is not None:
            return bc_type.lower() if isinstance(bc_type, str) else bc_type
        return None
    except ValueError:
        # Mixed BC - type property raises ValueError, try segments
        pass
    except AttributeError:
        # No .type attribute
        pass

    # Try segments for mixed BC
    try:
        from .types import BCType

        segments = boundary_conditions.segments
        if segments:
            first_type = segments[0].bc_type
            if isinstance(first_type, BCType):
                return first_type.value
            return str(first_type).lower()
    except (AttributeError, ImportError):
        pass

    # Legacy BC: direct attribute
    return getattr(boundary_conditions, "type", None)


def bc_type_to_geometric_operation(bc_type: str | None) -> str:
    """
    Map BC type string to geometric operation for Semi-Lagrangian solvers.

    Args:
        bc_type: BC type string from get_bc_type_string()

    Returns:
        Geometric operation: 'reflect', 'clamp', or 'periodic'

    Mapping:
        - 'periodic' → 'periodic' (wrap around domain)
        - 'neumann', 'no_flux', 'robin' → 'reflect' (mirror at boundary)
        - 'dirichlet', 'absorbing', None → 'clamp' (stay at boundary)

    Example:
        >>> bc_type_to_geometric_operation('no_flux')
        'reflect'
        >>> bc_type_to_geometric_operation('periodic')
        'periodic'
        >>> bc_type_to_geometric_operation('dirichlet')
        'clamp'
    """
    if bc_type is None:
        return "clamp"  # Default: absorbing

    bc_type_lower = bc_type.lower()

    if bc_type_lower == "periodic":
        return "periodic"
    elif bc_type_lower in ("neumann", "no_flux", "robin"):
        return "reflect"
    else:  # dirichlet, absorbing, or unknown
        return "clamp"


def geometric_operations(boundary_conditions: Any) -> set[str]:
    """Every distinct geometric operation ``boundary_conditions`` asks for.

    Unlike :func:`get_bc_type_string`, which returns the FIRST segment's type, this reports the
    whole set. A set of size > 1 means the BC cannot be honoured by a fold that applies one
    operation to every axis.

    ``default_bc`` is included deliberately. ``get_bc_type_string`` never reads it, so a
    partially-covering segment list plus a differing default produces the same silent collapse
    **with no permutation available** -- a guard that unions only over ``segments`` lets that form
    straight through (Issue #1697).

    Returns an empty set for ``None`` and for legacy BC objects, which carry neither field and so
    have no per-axis information that could disagree.

    Reach is by duck typing rather than ``isinstance`` on purpose. An ``isinstance`` gate would be
    a fail-silent branch in front of a fail-loud body: any future adapter, protocol implementation
    or wrapper that is not literally a ``BoundaryConditions`` would return an empty set, which
    reads as "nothing disagrees" and turns every caller's guard into a no-op -- the shape this
    function exists to prevent.

    Raises:
        AttributeError: if exactly one of ``segments`` / ``default_bc`` is present. That is the
            signature of a rename, and it must not degrade into an empty set (Issue #1691).
    """
    if boundary_conditions is None:
        return set()

    missing = object()
    segments = getattr(boundary_conditions, "segments", missing)
    default = getattr(boundary_conditions, "default_bc", missing)

    if segments is missing and default is missing:
        return set()  # not a segmented BC at all

    if segments is missing or default is missing:
        present, absent = ("segments", "default_bc") if default is missing else ("default_bc", "segments")
        raise AttributeError(
            f"{type(boundary_conditions).__name__} has {present!r} but no {absent!r}. A segmented "
            f"boundary condition must expose both, since a mixed BC is detected by unioning them; "
            f"reading only one would silently under-report disagreement (Issue #1697)."
        )

    def _op(bc_type: Any) -> str:
        return bc_type_to_geometric_operation(str(getattr(bc_type, "value", bc_type)))

    ops = {_op(seg.bc_type) for seg in segments or ()}
    if default is not None:
        ops.add(_op(default))
    return ops


def checked_bc_type_string(boundary_conditions: Any, *, consumer: str, alternative: str) -> str | None:
    """Collapse ``boundary_conditions`` to one BC type, or refuse if that would change the physics.

    The single owner of the per-axis collapse for solvers whose fold applies one geometric
    operation to every axis (Issues #1560, #1697). Callers get either a BC type they may safely
    apply to all axes, or ``NotImplementedError``.

    Call this at the point of use, not only at construction: solvers re-read
    ``get_boundary_conditions()`` at solve time, so a construction-time check alone is bypassed by
    a BC that is unset when the solver is built, or replaced afterwards.

    Args:
        boundary_conditions: the BC to collapse.
        consumer: the refusing component, named in the error (e.g. ``"HJBSemiLagrangianSolver"``).
        alternative: what the caller should use instead, appended to the error message.

    Per-axis handling is the actual fix and remains open on #1560 (HJB) and #1697 (FP). Until then
    the library refuses the configuration rather than solving a different one.
    """
    ops = geometric_operations(boundary_conditions)
    if len(ops) > 1:
        raise NotImplementedError(
            f"{consumer} does not support a mixed per-axis boundary condition whose segments map "
            f"to different geometric operations ({sorted(ops)}). The fold applies a single "
            "operation to every axis, so the result depends on segment order rather than on "
            f"which wall carries which condition. {alternative}"
        )
    return get_bc_type_string(boundary_conditions)


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    """Quick validation of BC utilities."""
    from .factories import dirichlet_bc, no_flux_bc, periodic_bc

    print("Testing BC utilities...")

    # Test no_flux
    bc_noflux = no_flux_bc(dimension=1)
    bc_type = get_bc_type_string(bc_noflux)
    assert bc_type == "no_flux"
    assert bc_type_to_geometric_operation(bc_type) == "reflect"
    print("  no_flux -> reflect: OK")

    # Test periodic
    bc_periodic = periodic_bc(dimension=1)
    bc_type = get_bc_type_string(bc_periodic)
    assert bc_type == "periodic"
    assert bc_type_to_geometric_operation(bc_type) == "periodic"
    print("  periodic -> periodic: OK")

    # Test dirichlet
    bc_dirichlet = dirichlet_bc(dimension=1, value=0.0)
    bc_type = get_bc_type_string(bc_dirichlet)
    assert bc_type == "dirichlet"
    assert bc_type_to_geometric_operation(bc_type) == "clamp"
    print("  dirichlet -> clamp: OK")

    # Test None -> clamp (default)
    assert get_bc_type_string(None) is None
    assert bc_type_to_geometric_operation(None) == "clamp"
    print("  None -> clamp: OK")

    print("\nAll BC utility tests passed!")
