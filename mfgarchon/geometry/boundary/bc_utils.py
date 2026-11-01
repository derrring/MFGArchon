"""
Centralized boundary condition utilities for all solver types.

Issue #702: Shared BC type detection and operation mapping for FDM, SL, GFDM, etc.

This module provides utilities that replace duplicated BC handling logic in:
- fp_fdm_time_stepping._get_bc_type()
- fp_semi_lagrangian_adjoint._axis_operations()

All solvers should import from this module for consistent BC handling.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from .types import BCType


def get_bc_type_string(boundary_conditions: Any) -> str | None:
    """
    Extract BC type string from any BoundaryConditions object.

    Supports:
    - Unified BoundaryConditions (conditions.py) with .type property: a uniform BC's one segment, read
      through the owner (:attr:`BoundaryConditions.uniform_segment`). Any other BoundaryConditions has no
      single type, and ``.type`` raises ``ValueError``: the first segment's type, which this returned
      until #2512 row B1, governs no particular face (#1697).
    - A legacy BC object: its ``.type`` attribute, or ``None``

    Args:
        boundary_conditions: Any BC object

    Returns:
        BC type string (e.g., "periodic", "dirichlet", "no_flux") or None

    Raises:
        ValueError: for a BoundaryConditions that is not uniform.

    Example:
        >>> from mfgarchon.geometry.boundary import no_flux_bc
        >>> bc = no_flux_bc(dimension=1)
        >>> get_bc_type_string(bc)
        'no_flux'
    """
    if boundary_conditions is None:
        return None

    # A BoundaryConditions's `.type` raises ValueError unless it is uniform; that raise is the answer, not a
    # signal to fall back on a segment.
    bc_type = getattr(boundary_conditions, "type", None)
    if bc_type is None:
        return None
    return bc_type.lower() if isinstance(bc_type, str) else bc_type


#: Each BC type string the semi-Lagrangian fold has a reading for, and its geometric operation. Total: a string
#: not here is refused by name rather than read as a stop at the wall, which it used to be (#2292).
_GEOMETRIC_OPERATION = MappingProxyType(
    {
        "periodic": "periodic",  # wrap around the domain
        "neumann": "reflect",  # mirror at the boundary
        "no_flux": "reflect",
        "robin": "reflect",
        "reflecting": "reflect",  # the particle vocabulary's fold, which the fallback read as a stop
        "dirichlet": "clamp",  # stay at the boundary
        "absorbing": "clamp",  # the particle vocabulary's exit
    }
)

#: BC types with no geometric operation, what each is, and why neither a fold nor a stop at the wall imposes it.
_NO_GEOMETRIC_OPERATION = MappingProxyType(
    {
        "flux": (
            "a prescribed mass flux J.n = q",
            "it is a Fokker-Planck solver's own BC, read by FP-FDM's implicit step and by FP-FEM (#2581)",
        ),
        "extrapolation_linear": (
            "a ghost-value extrapolation",
            "it constrains the value's derivative at the wall (#2292)",
        ),
        "extrapolation_quadratic": (
            "a ghost-value extrapolation",
            "it constrains the value's derivative at the wall (#2292)",
        ),
    }
)


def bc_type_to_geometric_operation(bc_type: str | BCType | None) -> str:
    """
    Map BC type string to geometric operation for Semi-Lagrangian solvers.

    Args:
        bc_type: BC type string from get_bc_type_string(), or a BCType member, which is read by its value

    Returns:
        Geometric operation: 'reflect', 'clamp', or 'periodic'

    Mapping:
        - 'periodic' → 'periodic' (wrap around domain)
        - 'neumann', 'no_flux', 'robin', 'reflecting' → 'reflect' (mirror at boundary)
        - 'dirichlet', 'absorbing', None → 'clamp' (stay at boundary)

    Raises:
        NotImplementedError: for 'flux' and the extrapolations, which have no geometric operation.
        ValueError: for any other string, a graph type's "source" or "custom" among them (#2292).
        TypeError: for a member of any other enum, a GraphBCType among them, and for anything that is neither a string,
            a BCType nor None (#2292).

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
    if isinstance(bc_type, Enum):
        # Read by its value only as a BCType: a GraphBCType shares "dirichlet" and "neumann" with BCType while being a
        # graph node's condition, and another enum is no BC type at all (#2292, row N1).
        from .types import refuse_a_non_continuum_bc_type

        refuse_a_non_continuum_bc_type(bc_type, "bc_type_to_geometric_operation", "bc_type")
        bc_type = bc_type.value
    if not isinstance(bc_type, str):
        raise TypeError(
            f"bc_type_to_geometric_operation: {bc_type!r} is not a BC type string, a BCType or None (#2292)."
        )
    bc_type_lower = bc_type.lower()
    if bc_type_lower in _GEOMETRIC_OPERATION:
        return _GEOMETRIC_OPERATION[bc_type_lower]
    if bc_type_lower in _NO_GEOMETRIC_OPERATION:
        what, why = _NO_GEOMETRIC_OPERATION[bc_type_lower]
        raise NotImplementedError(
            f"bc_type_to_geometric_operation: the {bc_type_lower.upper()} boundary ({what}) has no geometric operation; "
            f"{why}."
        )
    raise ValueError(
        f"bc_type_to_geometric_operation: {bc_type!r} is no BC type this reads; it reads "
        f"{sorted(_GEOMETRIC_OPERATION)} and None. A graph type ('source', 'custom') has no continuum reading (#2292)."
    )


def geometric_operations(boundary_conditions: Any) -> set[str]:
    """Every distinct geometric operation ``boundary_conditions`` asks for.

    Unlike :func:`get_bc_type_string`, which answers only for a uniform BC, this reports the whole
    set. A set of size > 1 means the BC cannot be honoured by a fold that applies one operation to
    every axis.

    ``default_bc`` is included deliberately: a partially-covering segment list plus a differing
    default asks for two operations **with no permutation available** -- a guard that unions only
    over ``segments`` lets that form straight through (Issue #1697). The exception is a uniform BC: its one unrestricted segment
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


def per_axis_operations(
    boundary_conditions: Any, dimension: int, *, consumer: str, geometry: Any = None
) -> tuple[str, ...]:
    """The geometric operation on each axis, read face by face from the owner. #1560, #1697, #2512 row B1.

    The per-axis owner for the semi-Lagrangian pair: axis ``d``'s operation is
    :func:`bc_type_to_geometric_operation` of the segment :meth:`BoundaryConditions.segment_for_face`
    gives each of its two faces. No-flux on one axis and periodic on another is then two operations, one
    per axis, instead of the first segment's applied to both. Where a segment covers part of a face, the
    owner accepts it only if it asks for the same operation as the segment governing the rest of the face:
    this consumer applies one operation per face, so a sub-face segment changes nothing only then.

    ``geometry`` is the grid the consumer solves on, which tells the owner how far a ``region`` or
    ``sdf_region`` placement covers each face.

    ``None`` and a legacy BC without ``segments`` carry no per-face information: their one operation, from
    :func:`get_bc_type_string`, applies to every axis.

    Raises:
        NotImplementedError: when an axis's two faces ask for different operations. A fold has one rule per
            axis, so periodic on one face only, or reflect on one and clamp on the other, is refused rather
            than resolved by whichever face is read first.
        ValueError: from the owner, for a placement it cannot represent on a face (#2512 row B1).
    """
    from .types import BoundaryFace

    if boundary_conditions is None or getattr(boundary_conditions, "segments", None) is None:
        return (bc_type_to_geometric_operation(get_bc_type_string(boundary_conditions)),) * dimension
    # Refuses the #1691 rename signature -- `segments` with `default_bc` renamed away -- by name, before
    # the per-face read would fail on it for a different reason (#2284).
    geometric_operations(boundary_conditions)

    def operation(segment: Any) -> str:
        return bc_type_to_geometric_operation(str(segment.bc_type.value))

    def same_operation(part: Any, governing: Any) -> bool:
        return operation(part) == operation(governing)

    operations = []
    for axis in range(dimension):
        ops = [
            operation(
                boundary_conditions.segment_for_face(BoundaryFace(axis, side), geometry, equivalent=same_operation)
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


def _describe_bc_value(value: Any) -> object | None:
    """A description if this BC value is not verifiably zero, else None. One owner for that question,
    read by `describe_inhomogeneous_bc_data` and `fp_view_of_shared_bc`.

    ``float()`` is reached only for things it accepts: an array or a provider would otherwise raise
    TypeError out of a capability gate, and an all-zero array is a legitimate ``g = 0`` that must not
    crash. A complex value, zero included, is not verifiably zero: a boundary datum is real, and ``float()``
    on a NumPy complex scalar drops the imaginary part, which read ``np.complex128(1j)`` as zero. The value
    owner (`types.boundary_value`) refuses complex too, so the two give one verdict (#2512 row B2).
    """
    import numpy as np

    from mfgarchon.geometry.boundary.providers import is_provider

    if value is None:
        return None
    if is_provider(value):
        return "<provider>"
    if callable(value):
        return "<callable>"
    if np.iscomplexobj(value):
        return f"<complex {value!r}>"
    if isinstance(value, np.ndarray):
        return None if not value.any() else "<array>"
    try:
        return None if float(value) == 0.0 else float(value)
    except (TypeError, ValueError):
        return f"<unrecognised {type(value).__name__}>"


def fp_view_of_shared_bc(boundary_conditions: Any, *, consumer: str) -> Any:
    """The Fokker-Planck reading of a shared problem/geometry BC (#2512, row B3).

    One owner for the translation. A shared BC carries one datum per face, and two equations read it.
    The shared value is the HJB's: for a Dirichlet the exit cost, for a Neumann du/dn = g, the boundary
    cost per unit local time. On the FP side:

    - every DIRICHLET segment, and a DIRICHLET ``default_bc``, comes back with value 0: an absorbing
      wall, m = 0, its value dropped **by design** (ruled 2026-10-07);
    - a NEUMANN with g = 0, as a segment or the ``default_bc``, comes back as NO_FLUX with value 0: the
      reflecting pairing, zero total flux J.n = 0 (ruled 2026-10-08);
    - a NEUMANN whose g is not provably zero is **refused** (user ruling 2026-10-09). Its g is the HJB's
      boundary cost and says nothing about the agents' mass at the wall, and a mass flux through the
      wall is not a Neumann condition, so the two equations' BCs are specified separately. "Not provably
      zero" is `describe_inhomogeneous_bc_data`'s answer, so a callable or a provider is refused rather
      than assumed zero, and the ``default_bc`` fall-through is checked too (#1686);
    - a ROBIN, as a segment or the ``default_bc``, is **refused**, whatever its coefficients (user ruling
      2026-10-10). On the shared BC it is the HJB's alpha*u + beta*du/dn = g. An FP Robin condition is on
      the total flux, J.n = (D/beta)(alpha*m - g), which is not alpha*m + beta*dm/dn = g where the drift
      crosses the wall, so the HJB's coefficients do not say what the FP's are.

    Everything else is returned unchanged: NO_FLUX, REFLECTING and PERIODIC mean the same to both
    equations, and the EXTRAPOLATION types reach the FP solver's own support check, which refuses them.

    ``consumer`` names the solver in the refusal, which advises giving that solver its own
    ``boundary_conditions``.

    Only a BC the FP solver reads from the shared problem / geometry goes through here -- in a coupled
    solve or a standalone FP solve alike, since the solver cannot tell them apart. A BC passed to an FP
    solver explicitly is the FP's own: there ``DIRICHLET(g)`` is a prescribed density m = g, a ROBIN is
    the FP's total-flux Robin, and a NEUMANN is refused (`refuse_explicit_fp_neumann`).

    Anything that is not a ``BoundaryConditions`` (``None``, a graph geometry's node-BC config) is returned as is.
    """
    from dataclasses import replace

    from .conditions import BoundaryConditions
    from .types import BCType

    if not isinstance(boundary_conditions, BoundaryConditions):
        return boundary_conditions
    if values := describe_inhomogeneous_bc_data(boundary_conditions, bc_types={BCType.NEUMANN}):
        _refuse_a_shared_neumann_value(boundary_conditions, values, consumer)
    if any(seg.bc_type == BCType.ROBIN for seg in boundary_conditions.segments) or (
        boundary_conditions.default_bc == BCType.ROBIN
    ):
        _refuse_a_shared_robin(boundary_conditions, consumer)

    def fp_segment(seg: Any) -> Any:
        if seg.bc_type == BCType.NEUMANN:
            return replace(seg, bc_type=BCType.NO_FLUX, value=0.0)
        if seg.bc_type == BCType.DIRICHLET and _describe_bc_value(seg.value) is not None:
            return replace(seg, value=0.0)
        return seg

    segments = [fp_segment(seg) for seg in boundary_conditions.segments]
    changes: dict[str, Any] = {}
    if any(new is not old for new, old in zip(segments, boundary_conditions.segments, strict=True)):
        changes["segments"] = segments
    default = boundary_conditions.default_bc
    if default == BCType.NEUMANN:
        changes.update(default_bc=BCType.NO_FLUX, default_value=0.0)
    elif default == BCType.DIRICHLET and _describe_bc_value(boundary_conditions.default_value) is not None:
        changes["default_value"] = 0.0
    if not changes:
        return boundary_conditions  # nothing to translate: the caller keeps the geometry's own object
    return replace(boundary_conditions, **changes)


def _separate_bc_advice(consumer: str, kept: str) -> str:
    """How to specify the two equations' BCs separately, which both shared refusals advise."""
    return (
        f"keep {kept} on the shared BC for the HJB, and pass {consumer} "
        "boundary_conditions=no_flux_bc(dimension=...) for reflected agents; through problem.solve, "
        "build both solvers yourself and pass them as hjb_solver= and fp_solver="
    )


def _refuse_a_shared_neumann_value(boundary_conditions: Any, values: list[object], consumer: str) -> NoReturn:
    """The refusal of a shared NEUMANN whose g is not provably zero, naming both readings (#2512, row B3)."""
    from .types import BCType

    where = [
        f"segment {seg.name!r}"
        for seg in boundary_conditions.segments
        if seg.bc_type == BCType.NEUMANN and _describe_bc_value(seg.value) is not None
    ]
    if (
        boundary_conditions.default_bc == BCType.NEUMANN
        and _describe_bc_value(boundary_conditions.default_value) is not None
    ):
        where.append("default_bc=NEUMANN, the fall-through for faces no segment names")
    raise NotImplementedError(
        f"{consumer}: the problem's shared boundary condition has a NEUMANN value that is not zero "
        f"({', '.join(map(str, values))}, at {'; '.join(where)}). On the shared BC that value is the HJB's "
        "boundary cost, du/dn = g. It says nothing about the agents' mass at the wall, and a mass flux "
        "through the wall is not a Neumann condition, so the FP cannot read one from it (#2512, user ruling "
        f"2026-10-09). Specify the two equations' BCs separately: {_separate_bc_advice(consumer, 'NEUMANN(g)')}."
    )


def _refuse_a_shared_robin(boundary_conditions: Any, consumer: str) -> NoReturn:
    """The refusal of a shared ROBIN at the FP, naming both readings (#2512, user ruling 2026-10-10)."""
    from .types import BCType

    where = [f"segment {seg.name!r}" for seg in boundary_conditions.segments if seg.bc_type == BCType.ROBIN]
    if boundary_conditions.default_bc == BCType.ROBIN:
        where.append("default_bc=ROBIN, the fall-through for faces no segment names")
    raise NotImplementedError(
        f"{consumer}: the problem's shared boundary condition has a ROBIN condition (at {'; '.join(where)}). "
        "On the shared BC it is the HJB's, alpha*u + beta*du/dn = g. An FP Robin condition is on the total "
        "flux, J.n = (D/beta)(alpha*m - g), which is not alpha*m + beta*dm/dn = g where the drift crosses the "
        "wall, so the HJB's coefficients do not say what the FP's are (#2512, user ruling 2026-10-10). "
        f"Specify the two equations' BCs separately: {_separate_bc_advice(consumer, 'ROBIN')}."
    )


def refuse_shared_flux(boundary_conditions: Any, consumer: str) -> Any:
    """Refuse a FLUX in the BC the HJB and the FP of a problem share, and return the BC otherwise (#2581, B-2).

    A FLUX is a prescribed mass flux through the wall, ``J.n = q``: a condition on the Fokker-Planck equation alone,
    with no meaning for the HJB. The two equations' BCs are specified separately (user ruling 2026-10-09), so a BC
    on the problem or its geometry, which both read, carries none: every FLUX segment or ``default_bc`` in it is
    refused. An FP solver takes a FLUX as its own ``boundary_conditions``, a standalone FP solve included. Anything
    that is not a ``BoundaryConditions`` (``None``, a graph geometry's node BCs) is returned as is.
    """
    from .conditions import BoundaryConditions
    from .types import BCType

    if not isinstance(boundary_conditions, BoundaryConditions):
        return boundary_conditions
    where = (
        [f"segment(s) {names}"]
        if (names := [seg.name for seg in boundary_conditions.segments if seg.bc_type == BCType.FLUX])
        else []
    )
    if boundary_conditions.default_bc == BCType.FLUX:
        where.append("default_bc=FLUX")
    if where:
        raise ValueError(
            f"{consumer}: a FLUX boundary condition (a prescribed mass flux J.n = q) is the Fokker-Planck solver's "
            f"alone, and a BC on the problem or its geometry is the one the HJB and the FP share (#2581, user ruling "
            f"2026-10-09, B-2). Found: {'; '.join(where)}. Keep the HJB's condition on the geometry and pass "
            "flux_bc(...) to the FP solver as its own boundary_conditions; a standalone FP solve takes it there too."
        )
    return boundary_conditions


def refuse_explicit_fp_neumann(boundary_conditions: Any, consumer: str) -> Any:
    """Refuse a NEUMANN in a BC handed to an FP solver, and return the BC otherwise (#2512, row B3).

    A BC passed to an FP solver is the FP's own, and its NEUMANN would mean dm/dn = g. No FP solver
    implements that: under a drift the zero-flux wall they build is J.n = 0, which is not dm/dn = 0. So
    every NEUMANN segment or ``default_bc`` in it is refused, g = 0 included (ruled 2026-10-08). Anything
    that is not a ``BoundaryConditions`` is returned as is.
    """
    from .conditions import BoundaryConditions
    from .types import BCType

    if not isinstance(boundary_conditions, BoundaryConditions):
        return boundary_conditions
    where = (
        [f"segment(s) {names}"]
        if (names := [seg.name for seg in boundary_conditions.segments if seg.bc_type == BCType.NEUMANN])
        else []
    )
    if boundary_conditions.default_bc == BCType.NEUMANN:
        # The fall-through is refused even where no face reaches it; narrowing it to a reachable default
        # needs the face resolution of #2512's row B1.
        where.append(
            "default_bc=NEUMANN, the fall-through for faces no segment names (mixed_bc sets it unless told "
            "otherwise); pass default_bc=BCType.NO_FLUX, or build BoundaryConditions(segments=...) directly"
        )
    if where:
        raise NotImplementedError(
            f"{consumer}: a NEUMANN boundary condition passed to an FP solver means dm/dn = g, which no FP "
            "solver implements: under a drift, the zero-flux wall an FP solver builds, where it builds one, is "
            "J.n = 0, a different "
            f"condition (#2512). Found: {'; '.join(where)}. Use NO_FLUX (no_flux_bc()) for a reflecting "
            "wall. A NEUMANN on the problem's shared BC is the HJB's du/dn = g: the FP reads g = 0 as zero flux and "
            "refuses a nonzero g."
        )
    return boundary_conditions


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
            neither ``segments`` nor ``default_bc`` is not one (``None``, a graph's node-BC
            config) and yields ``[]``; carrying exactly one of the pair is malformed
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
    missing = object()
    segments = getattr(boundary_conditions, "segments", missing)
    default_bc = getattr(boundary_conditions, "default_bc", missing)
    if segments is missing and default_bc is missing:
        return []  # not a BoundaryConditions at all (None, a graph's node-BC config)
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
        described = _describe_bc_value(getattr(seg, "value", None))
        if described is not None:
            found.append(described)

    if default_bc is not None and (bc_types is None or default_bc in bc_types):
        described = _describe_bc_value(getattr(boundary_conditions, "default_value", None))
        if described is not None:
            found.append(described)

    return sorted(set(found), key=repr)
