"""
Centralized boundary condition utilities for all solver types.

Issue #702: Shared BC type detection and operation mapping for FDM, SL, GFDM, etc.

This module provides utilities that replace duplicated BC handling logic in:
- fp_fdm_time_stepping._get_bc_type()
- fp_semi_lagrangian_adjoint._axis_operations()
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
    straight through (Issue #1697). The exception is a uniform BC: its one unrestricted segment
    covers every face, so the face-level readers (`get_bc_type_at_boundary`, the FDM ghost path)
    never reach its default, and counting it made one periodic segment over a NO_FLUX default read as
    mixed (Issue #2472). Some per-point readers do consult a uniform BC's default; that is #2490.

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
    if default is not None and not getattr(boundary_conditions, "is_uniform", False):
        ops.add(_op(default))
    return ops


def declares_periodic(boundary_conditions: Any) -> bool:
    """Whether a face of ``boundary_conditions`` can be periodic: a periodic segment, or a periodic ``default_bc``.

    The one predicate both periodic-convention binders read (#1822): ``TensorProductGrid`` when a BC is
    attached to it, and ``BaseNumericalSolver`` for a BC the caller hands the solver. Both read segments
    only until #1560, so a seam left to the default reached the solvers with no convention.
    """
    from .types import BCType

    segments = getattr(boundary_conditions, "segments", None) or ()
    if any(getattr(seg, "bc_type", None) is BCType.PERIODIC for seg in segments):
        return True
    return getattr(boundary_conditions, "default_bc", None) is BCType.PERIODIC


def refuse_unplaceable_segments(boundary_conditions: Any, dimension: int, *, consumer: str) -> None:
    """Refuse, in a mix of operations, a segment the face reader cannot place. #2467, #1953, #2490.

    ``get_bc_type_at_boundary`` answers one type per face, and four kinds of segment make that answer
    something other than what the BC says: one with no ``boundary``, which it applies to every face --
    ``sdf_region``, ``normal_direction`` and ``region_name`` segments are all of this kind, since none
    of them can carry a ``boundary``; one with ``boundary="all"``, which it applies to no face; one whose
    ``boundary`` names no face of this domain, which it drops; and one whose ``region`` restricts it to
    part of a face, which it stretches over the whole face. A uniform BC is not affected, nor one asking for
    a single operation with a ``default_bc``: every face then reads that operation whatever the reader
    does. Without a default it is: a face no segment truly reaches has no condition at all, and the face
    reader would hand it the misplaced segment's -- a periodic one placed by ``normal_direction`` then
    wraps every face, where HJB-FDM's ghosts find no BC and raise.

    One owner for every consumer that reads a BC face by face: the semi-Lagrangian pair through
    :func:`per_axis_operations`, and FP-FDM's periodicity (#2495).
    """
    if boundary_conditions is None or getattr(boundary_conditions, "segments", None) is None:
        return
    mixed = len(geometric_operations(boundary_conditions)) > 1
    if boundary_conditions.is_uniform or (not mixed and getattr(boundary_conditions, "default_bc", None) is not None):
        return
    where = "in a mix of geometric operations" if mixed else "with no `default_bc` for the faces they miss"
    unbounded = [seg.name for seg in boundary_conditions.segments if seg.boundary is None]
    if unbounded:
        raise NotImplementedError(
            f"{consumer}: segments {unbounded} have no `boundary` {where}, so "
            "the faces they cover cannot be read per axis. Give each one the face it lies on with "
            "`boundary=` (#2467)."
        )
    everywhere = [seg.name for seg in boundary_conditions.segments if seg.boundary == "all"]
    if everywhere:
        raise NotImplementedError(
            f"{consumer}: segments {everywhere} use boundary='all' {where}, and "
            "the face reader applies 'all' to no face (#1953), so it cannot be read per axis. Name the "
            "faces each one covers."
        )
    from .types import parse_boundary_face

    nowhere = [
        f"{seg.name} ({type(seg.boundary).__name__} {seg.boundary!r})"
        for seg in boundary_conditions.segments
        if (face := parse_boundary_face(seg.boundary)) is None or not 0 <= face.axis < dimension
    ]
    if nowhere:
        raise NotImplementedError(
            f"{consumer}: segments {nowhere} name no face of this {dimension}-D domain {where}, "
            "so the face reader drops them and the default takes their place. Name "
            "a face such as 'x_min' or 'axis0_max'."
        )
    partial = [seg.name for seg in boundary_conditions.segments if seg.region is not None]
    if partial:
        raise NotImplementedError(
            f"{consumer}: segments {partial} cover part of a face (`region`) {where}, "
            "and the face reader would "
            "give each of them its whole face (#2490)."
        )


def per_axis_operations(boundary_conditions: Any, dimension: int, *, consumer: str) -> tuple[str, ...]:
    """The geometric operation on each axis, read face by face. #1560, #1697.

    The per-axis owner for the semi-Lagrangian pair: axis ``d``'s operation is
    :func:`bc_type_to_geometric_operation` of ``get_bc_type_at_boundary("axis<d>_min")`` and of
    ``"axis<d>_max"``. No-flux on one axis and periodic on another is then two operations, one per axis,
    instead of the first segment's applied to both.

    ``None`` and a legacy BC without ``segments`` carry no per-face information: their one operation, from
    :func:`get_bc_type_string`, applies to every axis.

    Raises:
        NotImplementedError: when an axis's two faces ask for different operations. A fold has one rule per
            axis, so periodic on one face only, or reflect on one and clamp on the other, is refused rather
            than resolved by whichever face is read first. Also, in a mix of operations, for a segment the
            face reader cannot place: one with no ``boundary``, whose faces cannot be told apart (#2467) --
            ``sdf_region``, ``normal_direction`` and ``region_name`` segments are all of this kind, since
            none of them can carry a ``boundary``; one with ``boundary="all"``, which the face reader
            applies to no face (#1953); one whose ``boundary`` names no face of this domain -- a misspelt
            name, a Gmsh tag, an axis past its dimension -- which the face reader drops and the default
            replaces; and one whose ``region`` restricts it to part of a face, which the face reader
            stretches over the whole face (#2490).
    """
    if boundary_conditions is None or getattr(boundary_conditions, "segments", None) is None:
        return (bc_type_to_geometric_operation(get_bc_type_string(boundary_conditions)),) * dimension
    refuse_unplaceable_segments(boundary_conditions, dimension, consumer=consumer)
    operations = []
    for axis in range(dimension):
        ops = [
            bc_type_to_geometric_operation(
                str(getattr(t, "value", t))
                if (t := boundary_conditions.get_bc_type_at_boundary(f"axis{axis}_{side}")) is not None
                else None
            )
            for side in ("min", "max")
        ]
        if ops[0] != ops[1]:
            raise NotImplementedError(
                f"{consumer}: axis {axis} asks for {ops[0]!r} at its min face and {ops[1]!r} at its max face. "
                "This consumer applies one operation per axis, so the two faces of an axis must agree."
            )
        operations.append(ops[0])
    return tuple(operations)


def per_axis_diffusion_types(operations: tuple[str, ...]) -> tuple[str, ...]:
    """The implicit diffusion's boundary on each axis: ``periodic`` where the fold wraps, ``neumann`` elsewhere."""
    return tuple("periodic" if op == "periodic" else "neumann" for op in operations)


def describe_inhomogeneous_bc_data(
    boundary_conditions: Any,
    *,
    bc_types: set[Any] | None = None,
) -> list[object]:
    """Values attached to ``boundary_conditions`` that are not verifiably zero.

    One owner for the question "is this boundary datum ``g = 0``?", asked by every guard
    that honours only the homogeneous case. Returns a sorted, de-duplicated list of
    descriptions, empty when every datum in scope is provably zero.

    Two channels, and both are load-bearing:

    - each entry in ``segments``;
    - the **fall-through** ``default_bc`` / ``default_value``, which is a value too. Issue
      #1686 recorded that checking only ``segments`` left the original silent discard
      reachable through ``default_bc=NEUMANN, default_value=g``, and #1802's first guard
      reproduced that hole 300 lines away, which is why this lives in one place now.

    Anything that cannot be compared to zero here -- a provider, a callable, an
    unrecognised type -- is **described rather than assumed homogeneous**, so the caller
    refuses it. ``isinstance(value, (int, float))`` alone would accept
    ``neumann_bc(value=lambda t: 5.0)`` and discard it silently, which is the behaviour
    these gates exist to stop.

    Args:
        boundary_conditions: a ``BoundaryConditions``-like object. Anything carrying
            neither ``segments`` nor ``default_bc`` is not one (a string sentinel,
            ``None``) and yields ``[]``; carrying exactly one of the pair is malformed
            and raises, matching ``geometric_operations`` (#1691) rather than degrading
            to "nothing disagrees".
        bc_types: restrict to segments/defaults of these ``BCType``\\ s. ``None`` means
            every type, which is what a caller whose transform breaks on ANY inhomogeneous
            condition wants. A caller that breaks on only ONE type passes that type --
            note the sense: callers pass the types they CANNOT honour, not the ones they
            can. ``base_solver`` passes ``{NEUMANN}`` precisely because
            ``honors_inhomogeneous_neumann`` is False, while it does honour a Dirichlet
            value. Reading this backwards inverts the gate.

    Returns:
        Descriptions of the offending values, e.g. ``[0.2]``, ``['<callable>']``.
    """
    import numpy as np

    from mfgarchon.geometry.boundary.providers import is_provider

    def _describe(value: Any) -> object | None:
        """A description if this value is not verifiably zero, else None.

        ``float()`` is reached only for things it accepts: an array or a provider would
        otherwise raise TypeError out of a capability gate, and an all-zero array is a
        legitimate ``g = 0`` that must not crash.
        """
        if value is None:
            return None
        if is_provider(value):
            return "<provider>"
        if callable(value):
            return "<callable>"
        if isinstance(value, np.ndarray):
            return None if not value.any() else "<array>"
        try:
            return None if float(value) == 0.0 else float(value)
        except (TypeError, ValueError):
            return f"<unrecognised {type(value).__name__}>"

    missing = object()
    segments = getattr(boundary_conditions, "segments", missing)
    default_bc = getattr(boundary_conditions, "default_bc", missing)
    if segments is missing and default_bc is missing:
        return []  # not a BoundaryConditions at all (None, a string sentinel)
    if segments is missing or default_bc is missing:
        # Issue #1691, same rule as geometric_operations in this module: an object
        # exposing one half of the pair is malformed, and answering "nothing disagrees"
        # for it would be a capability gate degrading into a pass.
        present, absent = ("default_bc", "segments") if segments is missing else ("segments", "default_bc")
        raise AttributeError(
            f"{type(boundary_conditions).__name__} has {present!r} but no {absent!r}. A segmented "
            "boundary condition must expose both; refusing to report it as homogeneous."
        )

    found: list[object] = []
    for seg in segments or ():
        if bc_types is not None and seg.bc_type not in bc_types:
            continue
        described = _describe(getattr(seg, "value", None))
        if described is not None:
            found.append(described)

    if default_bc is not None and (bc_types is None or default_bc in bc_types):
        described = _describe(getattr(boundary_conditions, "default_value", None))
        if described is not None:
            found.append(described)

    return sorted(set(found), key=repr)
