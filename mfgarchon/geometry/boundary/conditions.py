"""
Unified boundary condition specification (dimension-agnostic).

This module provides the canonical BoundaryConditions class supporting:
- **Uniform BCs**: Single segment covering all boundaries (same type everywhere)
- **Mixed BCs**: Multiple segments with different types on different boundaries
- **Rectangular domains**: Axis-aligned boundaries via `domain_bounds`
- **General/Lipschitz domains**: SDF-defined boundaries via `domain_sdf`

Use factory functions for convenient creation:
- `uniform_bc()`, `periodic_bc()`, `dirichlet_bc()`, etc. for uniform BCs
- `mixed_bc()` for mixed BCs with multiple segments

Examples:
    Uniform Neumann BC:
    >>> bc = neumann_bc(dimension=2)
    >>> assert bc.is_uniform

    Mixed BC with exit and walls:
    >>> from mfgarchon.geometry.boundary import BCSegment, BCType
    >>> exit_seg = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0,
    ...                      boundary="x_max", priority=1)
    >>> wall_seg = BCSegment(name="walls", bc_type=BCType.NEUMANN, value=0.0)
    >>> bc = mixed_bc([exit_seg, wall_seg], dimension=2,
    ...               domain_bounds=np.array([[0, 10], [0, 10]]))
    >>> assert bc.is_mixed
"""

from __future__ import annotations

import itertools
import weakref
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from mfgarchon.geometry.protocols import SupportsRegionMarking
from mfgarchon.utils.deprecation import deprecated

from .tolerances import BOUNDARY_REL_TOL, BOUNDARY_TOL, SDF_BOUNDARY_TOL
from .types import (
    BCSegment,
    BCType,
    BoundaryFace,
    PeriodicGridConvention,
    _compute_sdf_gradient,
    _normal_selects_face,
    normal_holds,
    parse_boundary_face,
    region_holds,
    repeated_endpoint_count,
    sdf_region_holds,
    segment_value,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, eq=False)
class BoundaryFacets:
    """A mesh's boundary facets, as `BoundaryConditions.segments_for_facets` reads them (#2512 row B1, stage B1b).

    ``midpoints`` and ``normals`` (outward, unit) hold one row per facet. ``tags`` maps each tag of the mesh to the
    rows of the boundary facets it holds; ``off_boundary`` names the tags that also hold facets inside the domain.
    """

    midpoints: np.ndarray
    normals: np.ndarray
    tags: dict[str, np.ndarray]
    off_boundary: frozenset[str] = frozenset()


@dataclass
class BoundaryConditions:
    """
    Unified boundary condition specification (uniform or mixed, any dimension).

    This is the canonical boundary condition class supporting:
    - **Uniform BCs**: Single segment covering all boundaries (same type everywhere)
    - **Mixed BCs**: Multiple segments with different types on different boundaries
    - **Rectangular domains**: Axis-aligned boundaries via `domain_bounds`
    - **General/Lipschitz domains**: SDF-defined boundaries via `domain_sdf`

    Use factory functions for convenient creation:
    - `uniform_bc()`, `periodic_bc()`, `dirichlet_bc()`, etc. for uniform BCs
    - `mixed_bc()` for mixed BCs with multiple segments

    Attributes:
        dimension: Spatial dimension of the problem (1, 2, 3, ...) or None for lazy binding.
            When None, dimension will be inferred when BC is attached to a Geometry.
        segments: List of BC segments (ordered by priority)
        default_bc: Fallback BC type for points/segments that match no explicit
            segment. ``None`` (the default) means *unspecified*: resolving a
            fall-through point then raises instead of silently applying a default
            (Issue #1100 — PERIODIC is never applied unless explicitly requested).
            Set explicitly (e.g. ``BCType.NO_FLUX``) when segments may not cover
            every boundary point.
        default_value: Default BC value when no segment matches
        domain_bounds: Domain bounds array of shape (dimension, 2) for rectangular domains
        domain_sdf: Signed distance function for general/Lipschitz domains
        corner_strategy: How to handle corners/edges ("priority", "average", "mollify")
        corner_mollification_radius: Smoothing radius for "mollify" strategy

    Examples:
        Uniform Neumann BC:
        >>> bc = neumann_bc(dimension=2)
        >>> assert bc.is_uniform

        Mixed BC with exit and walls:
        >>> exit_seg = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0,
        ...                      boundary="x_max", priority=1)
        >>> wall_seg = BCSegment(name="walls", bc_type=BCType.NEUMANN, value=0.0)
        >>> bc = mixed_bc([exit_seg, wall_seg], dimension=2,
        ...               domain_bounds=np.array([[0, 10], [0, 10]]))
        >>> assert bc.is_mixed

        Circular domain with exit at top (Lipschitz/SDF):
        >>> exit_seg = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0,
        ...                      normal_direction=np.array([0, 1]), priority=1)
        >>> bc = mixed_bc([exit_seg], dimension=2,
        ...               domain_sdf=lambda x: np.linalg.norm(x) - 5.0)
    """

    dimension: int | None = None
    segments: list[BCSegment] = field(default_factory=list)
    # Issue #1100: None = "unspecified" (no silent PERIODIC fallback). Resolving a
    # fall-through point with default_bc is None fails loud via _resolve_default_bc.
    default_bc: BCType | None = None
    default_value: float = 0.0

    # Rectangular domain specification
    domain_bounds: np.ndarray | None = None

    # General domain specification (SDF-based, supports Lipschitz boundaries)
    domain_sdf: Callable[[np.ndarray], float] | None = None

    # Corner handling (important for Lipschitz domains with re-entrant corners)
    corner_strategy: Literal["priority", "average", "mollify"] = "priority"
    corner_mollification_radius: float = 0.1
    # Issue #1822. Rides on the BC because a periodic wrap happens ONLY because this object says
    # PERIODIC, so this is the one carrier that reaches every wrapping site without threading a
    # new argument through the 31 that pad or construct.
    #
    # ``None`` means UNSTATED, and unstated wraps the way this package always wrapped -- N
    # distinct nodes -- so attaching this field changes no existing caller's numbers. Defaulting
    # it to the other convention instead was measured and reverted: it silently reinterprets every
    # BC that already meant exclusive, and took the operator layer's own Laplacian from 1.3e-02 to
    # 6.3e+02 with nothing red. A grid does not leave it unstated for long -- ``TensorProductGrid``
    # BINDS the convention it measures from its own coordinates when a periodic BC is attached, the
    # same way it binds ``dimension``, and refuses one that contradicts them.
    periodic_convention: PeriodicGridConvention | None = None
    # #2512 row B1: the geometry this BC is bound to, held weakly. It rides on the BC as `periodic_convention` does,
    # which #1822 put there so that "nothing had to be threaded through the 31 call sites that pad or construct": an
    # owner query made deep in an operator -- a ghost pad inside a Laplacian -- has no geometry argument. Written only
    # by `bind_geometry`, which a grid calls on the BC it stores. The owner reads it only when a query passes no
    # geometry; a reader that holds the geometry passes it. A query that passes none, on a BC bound to no geometry,
    # for a placement only a grid decides, is refused by name. Not compared and not shown.
    # Where this lives (on the BC, or held by the applicators and threaded) is revisited after #2512's B1e (user
    # direction, 2026-10-13).
    _geometry_ref: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Sort segments by priority (highest first), and refuse a face of an axis the domain lacks."""
        self.segments.sort(key=lambda seg: seg.priority, reverse=True)
        if self.dimension is not None:
            self._refuse_missing_axes(self.dimension)

    def _refuse_missing_axes(self, dimension: int) -> None:
        """Refuse a segment placed on a face of an axis a ``dimension``-D domain lacks: it can govern no location,
        whichever reader asks, and would be dropped (#2512 row B1, ruling (c)). Needs only the dimension, so it runs
        wherever the dimension becomes known -- construction, `bind_dimension` (a `replace`, so construction again),
        and a geometry handing it to an unbound BC -- rather than in one reader's path."""
        for segment in self.segments:
            if segment.boundary is None or segment.boundary == "all":
                continue
            named = parse_boundary_face(segment.boundary)
            if named is not None and not 0 <= named.axis < dimension:
                raise ValueError(
                    f"BC segment {segment.name!r} is placed on {segment.boundary!r}, a face of axis {named.axis}, "
                    f"and this domain is {dimension}-D, so it can govern no face: it would be dropped. "
                    "Name a face of this domain (#2512 row B1)."
                )

    # =========================================================================
    # Lazy dimension binding
    # =========================================================================

    @property
    def is_bound(self) -> bool:
        """Check if dimension has been bound (explicitly or via lazy binding)."""
        return self.dimension is not None

    def bind_dimension(self, dim: int) -> BoundaryConditions:
        """
        Bind dimension to this BC specification (lazy binding).

        Called by Geometry when BC is attached. If dimension is already set,
        validates consistency. Returns a new BoundaryConditions instance with
        the bound dimension.

        Args:
            dim: Spatial dimension to bind

        Returns:
            BoundaryConditions with dimension set

        Raises:
            ValueError: If BC already has a different dimension

        Example:
            >>> bc = dirichlet_bc(value=0.0)  # dimension=None
            >>> bc_2d = bc.bind_dimension(2)  # Now dimension=2
            >>> assert bc_2d.dimension == 2
        """
        if self.dimension is not None and self.dimension != dim:
            raise ValueError(
                f"BC dimension mismatch: BC has dimension={self.dimension}, but geometry has dimension={dim}. Build "
                f"the BC with dimension={dim}, the geometry's."
            )
        if self.dimension == dim:
            return self  # Already bound to correct dimension
        # A list of its own: __post_init__ sorts the segments in place, and the caller's list is not this copy's.
        return replace(self, dimension=dim, segments=list(self.segments))

    def bind_geometry(self, geometry: Any) -> BoundaryConditions:
        """This BC bound to ``geometry``: its dimension, and the geometry the owner decides a placement on when a reader
        asks without one (#2512 row B1). Returns ``self`` when it is already bound to this geometry, and otherwise a
        copy, so binding a BC that two solves share never redirects the other one's queries."""
        bound = self.bind_dimension(geometry.dimension)
        if bound._geometry_ref is not None and bound._geometry_ref() is geometry:
            return bound
        return replace(bound, segments=list(bound.segments), _geometry_ref=weakref.ref(geometry))

    def _query_geometry(self, geometry: Any) -> Any:
        """The geometry a query is decided on: the one passed, else the one this BC is bound to, if it is alive."""
        if geometry is not None or self._geometry_ref is None:
            return geometry
        return self._geometry_ref()

    def __getstate__(self) -> dict[str, Any]:
        """Pickle and copy this BC without the owner's two weak references: the binding (`_geometry_ref`) and the
        memo (`_owner_memo`). A weak reference cannot be pickled, so a problem holding a bound or queried BC could
        not cross a process pool (#2634, review 2); and a copy that kept the binding would decide its placements on
        the original's grid. The copy is unbound, and a grid that holds it binds it again (`TensorProductGrid`'s
        ``__setstate__``). A copy of the same BC held elsewhere -- deep-copied beside its grid, say -- stays unbound,
        and a query on it without a geometry is decided on ``domain_bounds``; the library holds a grid's BC only on
        the grid."""
        state = dict(self.__dict__)
        state["_geometry_ref"] = None
        state.pop("_b1_owner_cache", None)
        return state

    def _require_dimension(self, operation: str = "this operation") -> int:
        """
        Internal helper: require dimension to be bound before certain operations.

        Args:
            operation: Name of operation for error message

        Returns:
            Bound dimension

        Raises:
            ValueError: If dimension is not bound
        """
        if self.dimension is None:
            raise ValueError(
                f"BC dimension not set. Cannot perform {operation}. "
                f"Either specify dimension in factory function (e.g., dirichlet_bc(dimension=2)) "
                f"or attach BC to a Geometry to bind dimension automatically."
            )
        return self.dimension

    # =========================================================================
    # Properties to distinguish uniform vs mixed BCs
    # =========================================================================

    @property
    def is_uniform(self) -> bool:
        """
        Check if this is a uniform BC (single segment covering all boundaries).

        Uniform BCs have exactly one segment with no boundary restriction. A marked region
        (``region_name``) is a restriction: read as uniform, one region-named segment governed every
        face and ``default_bc`` was never consulted (#2472).
        """
        if len(self.segments) != 1:
            return False
        seg = self.segments[0]
        return (
            seg.boundary is None
            and seg.region is None
            and seg.sdf_region is None
            and seg.normal_direction is None
            and seg.region_name is None
        )

    @property
    def is_mixed(self) -> bool:
        """
        Check if this is a mixed BC (multiple segments or boundary-specific).

        Mixed BCs have multiple segments or segments targeting specific boundaries.
        """
        return not self.is_uniform

    @property
    def uniform_segment(self) -> BCSegment | None:
        """A segment whose condition every location of the boundary carries, or ``None`` (#2512 row B1).

        Decided by the owner's one sameness rule, `_same_condition` (type, value, flux capacity, Robin coefficients),
        and not by asking :meth:`segment_for_face` on each face: when every condition the BC can impose is the same,
        which segment governs a location cannot matter, so no placement is read, no geometry is needed, and a BC on
        a domain with no box faces is answered too. The conditions it can impose are every segment's, and
        ``default_bc``'s unless a global segment (no placement field, or ``boundary="all"``) covers every location,
        in which case the default governs none. With no global segment and no ``default_bc``, a location may fall
        through to the unset default, which the owner refuses where it is asked, so the answer is ``None``. ``None``
        sends a consumer down its per-face or per-point path, which reads the owner.

        A consumer whose fast path handles only a uniform BC reads this, never ``segments[0]``, which on any other
        BC is whichever segment was declared first and governs no particular location. Four alike segments, one
        per face, are uniform; so is a BC with no segments and a ``default_bc``.
        """
        candidates = list(self.segments)
        if not any(_is_global(segment) for segment in self.segments):
            if self._default_refusal("uniform_segment", "the boundary") is not None:
                return None  # unset, or a bare ROBIN with no alpha/beta: the owner refuses it where it governs
            candidates.append(self._default_segment("uniform_segment", "the boundary"))
        if not candidates:
            return None
        first = candidates[0]
        return first if all(_same_condition(segment, first) for segment in candidates[1:]) else None

    # =========================================================================
    # Dynamic BC Value Provider Support (Issue #625)
    # =========================================================================

    def has_providers(self) -> bool:
        """
        Check if any segment has a BCValueProvider value.

        Used by FixedPointIterator to determine if BC resolution is needed
        before passing to solvers.

        Returns:
            True if any segment carries a BCValueProvider in ``value``, ``alpha`` or ``beta``.

        Example:
            >>> if bc.has_providers():
            ...     bc = bc.with_resolved_providers(state)

        Note:
            This is the gate in front of ``with_resolved_providers``, whose fast path returns
            ``self`` unchanged when this is False. It must therefore cover exactly the fields
            that method resolves. ~~``seg.value`` only~~ [CORRECTED 2026-08-16]: widening the
            resolver without widening the gate would leave a provider on ``alpha`` silently
            unresolved -- the caller would receive the provider object itself where a number was
            expected, and the failure would surface far downstream as a type error in a ghost
            formula rather than here.
        """
        from .providers import is_provider

        return any(is_provider(getattr(seg, f)) for seg in self.segments for f in ("value", "alpha", "beta"))

    def with_resolved_providers(
        self,
        state: dict[str, Any],
    ) -> BoundaryConditions:
        """
        Create a new BoundaryConditions with all providers resolved to concrete values.

        This is the primary method for the FixedPointIterator to resolve dynamic
        BCs before passing them to solvers. Returns a new instance where all
        BCValueProvider values have been replaced with their computed float values.

        Args:
            state: Iteration state dict passed to provider.compute().
                   Standard keys: 'm_current', 'U_current', 'geometry', 'volatility'.

        Returns:
            New BoundaryConditions instance with concrete values (no providers)

        Example:
            >>> # In FixedPointIterator
            >>> if problem.boundary_conditions.has_providers():
            ...     resolved_bc = problem.boundary_conditions.with_resolved_providers(state)
            ... else:
            ...     resolved_bc = problem.boundary_conditions
            >>> U_new = hjb_solver.solve(bc=resolved_bc, ...)
        """
        from .providers import is_provider

        if not self.has_providers():
            return self  # Fast path: no providers to resolve

        resolved_segments = []
        for seg in self.segments:
            # `value`, `alpha` and `beta` may each carry a provider. The impermeable wall of a
            # Fokker-Planck equation is Robin with `alpha` the outward normal drift,
            # `D_pH(x, grad u) . n` -- a quantity that is only knowable from the current
            # iterate, which is what a provider is for, and which lives on `alpha`, not on
            # `value`. ~~only `value` was resolved~~ [CORRECTED 2026-08-16]
            updates = {
                field: getattr(seg, field).compute(state)
                for field in ("value", "alpha", "beta")
                if is_provider(getattr(seg, field))
            }
            resolved_segments.append(replace(seg, **updates) if updates else seg)

        return replace(self, segments=resolved_segments)

    @property
    def type(self) -> str:
        """
        Get the BC type string (for uniform BCs).

        For uniform BCs, returns the type string (e.g., "periodic", "dirichlet").
        For mixed BCs, raises ValueError - use segments directly.

        This property provides compatibility with code expecting the old
        BoundaryConditions.type attribute.
        """
        segment = self.uniform_segment
        if segment is None:
            raise ValueError(
                "type property only valid for uniform BCs. A mixed BC has no single type: read each face through "
                "segment_for_face (#2512 row B1)."
            )
        return segment.bc_type.value

    @property
    def bc_type(self) -> BCType:
        """
        Get the BCType enum (for uniform BCs).

        For uniform BCs, returns the BCType enum value.
        For mixed BCs, raises ValueError.
        """
        segment = self.uniform_segment
        if segment is None:
            raise ValueError(
                "bc_type property only valid for uniform BCs. A mixed BC has no single type: read each face "
                "through segment_for_face (#2512 row B1)."
            )
        return segment.bc_type

    def _resolve_default_bc(self, context: str) -> BCType:
        """Return ``default_bc`` for a fall-through point, or fail loud if unset.

        Issue #1100: ``default_bc`` is ``None`` when the user never specified a
        fallback. A point/segment that matches no explicit BC segment falls
        through to ``default_bc``; silently substituting a default (historically
        ``PERIODIC``) is wrong physics, so resolution raises here instead of
        guessing. PERIODIC is applied only when the user sets it explicitly.

        Args:
            context: Caller name, surfaced in the error for debuggability.

        Returns:
            The explicitly-set default ``BCType``.

        Raises:
            ValueError: If ``default_bc`` is ``None`` (unspecified).
        """
        if self.default_bc is None:
            raise ValueError(
                f"{context}: point/segment matched no BC and default_bc was not "
                "specified on BoundaryConditions; set default_bc=BCType.NO_FLUX / "
                "PERIODIC / DIRICHLET explicitly (Issue #1100)."
            )
        return self.default_bc

    # ------------------------------------------------------------------------------------------
    # Row B1 (#2512): which segment governs a boundary location. One placement rule, two questions:
    # a face (readers that impose one condition per face) and a point (readers that resolve per point).
    # Every reader asks here; none places a segment by its own reading of the placement fields.
    # ------------------------------------------------------------------------------------------

    def segment_for_face(
        self,
        face: BoundaryFace | str,
        geometry: Any = None,
        equivalent: Callable[[BCSegment, BCSegment], bool] | None = None,
    ) -> BCSegment:
        """The segment that governs ``face`` where one condition is imposed per face (#2512 row B1).

        **The order on a face**, written here and nowhere else: (1) ``priority``, descending; (2) then
        specificity, with two levels -- a segment with no placement field or with ``boundary="all"`` is
        GLOBAL ("the rest": ``default_bc`` is a bare type and cannot carry a value or Robin coefficients,
        so an unplaced segment is how a BC says "everywhere else, with this condition"), any other is
        PLACED, and placed beats global; (3) then declaration order. The first segment in that order
        that governs the WHOLE face wins. **At a corner** (a node or point on several faces,
        :meth:`segment_for_corner`) one more rule enters after priority: essential before natural --
        DIRICHLET before any other condition -- so that a corner where a Dirichlet face meets another at
        equal priority is the Dirichlet face's, whatever order the faces were declared in. Faces themselves
        take no account of type:

        - ``boundary``: that face, aliases included (``parse_boundary_face``); ``"all"`` is every face.
        - ``region`` (alone, or with ``boundary``) and ``sdf_region``: the face if the placement holds
          its whole interior -- read on ``geometry``'s grid nodes when it is a grid, else (``region``
          only) on ``domain_bounds``. ``region_name``: the geometry's mask, or a face label
          (:func:`region_name_governs_face`).
        - ``normal_direction``: the face whose outward normal is within ``normal_tolerance`` of it.
        - no placement field: every face.

        A segment that covers PART of the face at a higher priority than the one governing the rest is
        refused by name unless the two impose the same condition -- ``equivalent(part, governing)``,
        by default the same type, value and Robin coefficients -- because one condition per face
        cannot represent both, and either reading would silently drop the other (#2472, #2490). A
        consumer that needs less than the whole condition (an operation, say) passes its own
        ``equivalent``. A placement this reader cannot decide (a ``boundary`` naming no box face, an
        ``sdf_region`` with no grid) is refused. Nothing governing the face gives ``default_bc``
        (:meth:`_default_segment`), which is refused when unset or ROBIN.

        ``face`` may also be a name that is no box face (a mesh tag): a segment then governs it by
        equal ``boundary`` or ``region_name``, or by carrying no placement at all.
        """
        target = parse_boundary_face(face)
        if target is None:
            return self._segment_for_tag(face)
        geometry = self._query_geometry(geometry)
        governing, part = self._face_resolution(target, geometry, equivalent, "segment_for_face")
        if part is None:
            return governing
        if governing is None:  # nothing covers the face whole, and the default does not answer: its refusal
            raise ValueError(self._default_refusal("segment_for_face", f"the {target.to_string()} face"))
        how = "region" if part.region is not None else "sdf_region" if part.sdf_region is not None else "placement"
        raise ValueError(
            f"BC segment {part.name!r} covers part of the {target.to_string()} face (placed by `{how}`) "
            f"at a higher priority than {governing.name!r}, which governs the rest of it, and the two "
            "impose different conditions. A face-level boundary condition imposes one condition per "
            "face and cannot represent both: either reading would silently drop the other. Cover whole "
            "faces, or use a solver that resolves boundary conditions per point (#2512 row B1, #2490)."
        )

    def _face_resolution(
        self,
        target: BoundaryFace,
        geometry: Any,
        equivalent: Callable[[BCSegment, BCSegment], bool] | None,
        context: str,
    ) -> tuple[BCSegment | None, BCSegment | None]:
        """The face owner's steps, in one place (#2512 row B1): :meth:`segment_for_face` refuses on their outcome,
        and a point at a corner (:meth:`segment_for_point`) reads it.

        The whole-BC check (:meth:`_refuse_shadowed_segments`) runs first, and its refusals raise. Then, in the face
        order, the first segment covering the face whole governs it -- or, with none, ``default_bc``. Returns
        ``(governing, part)``:

        - ``(segment, None)``: the face answers ``segment``. Every partial cover ahead of it imposes its condition
          (``equivalent``, by default `_same_condition`).
        - ``(governing, part)``: ``part`` covers part of the face ahead of ``governing`` and imposes another
          condition, so one condition for the face cannot represent both.
        - ``(None, part)``: no segment covers the face whole and the default does not answer (unset or ROBIN), and
          the partial cover ``part`` is why no segment governs it.

        A face no segment covers at all, with a default that does not answer, raises the default's refusal; so does
        a cover that cannot be decided. Only a partial cover is returned rather than raised.
        """
        same = _same_condition if equivalent is None else equivalent
        self._refuse_shadowed_segments(target, geometry, equivalent)
        partial: list[BCSegment] = []
        governing: BCSegment | None = None
        for segment in self._face_order():
            cover = self._cached_face_cover(segment, target, geometry)
            if cover == "whole":
                governing = segment
                break
            if cover == "partial":
                partial.append(segment)
        if governing is None:
            if partial and self._default_refusal(context, f"the {target.to_string()} face") is not None:
                return None, partial[0]
            governing = self._default_segment(context, f"the {target.to_string()} face")
        return governing, next((part for part in partial if not same(part, governing)), None)

    def placed_segments(self) -> list[BCSegment]:
        """The segments placed by anything but a face label: by ``region``, ``sdf_region``, ``region_name`` or
        ``normal_direction`` (#2490). A reader that resolves whole faces by label alone sends a BC carrying one down its
        per-point path. An unmarked ``region_name`` such as ``"x_max"`` and an axis-aligned ``normal_direction`` count,
        conservatively, although each decides a whole face (:meth:`segments_placed_off_face_labels`)."""
        return [
            seg
            for seg in self.segments
            if seg.region is not None
            or seg.sdf_region is not None
            or seg.region_name is not None
            or seg.normal_direction is not None
        ]

    def segments_placed_off_face_labels(self) -> list[BCSegment]:
        """The segments a reader that integrates over whole box faces cannot place where it can matter to a DIRICHLET
        condition (#2490, #2717): those placed by ``region``, by ``sdf_region``, or by a ``region_name`` that the bound
        grid marks or that is no face label, and that are DIRICHLET, come before a DIRICHLET segment in the face order,
        or sit beside a DIRICHLET ``default_bc``.

        A face label -- ``boundary``, an unmarked ``region_name`` such as ``"x_max"``, or ``"all"`` -- and
        ``normal_direction`` on an axis-aligned face decide a whole face exactly. A ``region``, an ``sdf_region`` and a
        marked region, whatever its name, are decided on the grid's nodes (:meth:`segment_for_face`,
        `region_name_governs_face`), which need not be the faces such a reader integrates; placing a condition on part
        of a face is per-point resolution (#2712).
        """
        order = self._face_order()
        geometry = self._query_geometry(None)
        dirichlet_default = self.default_bc is not None and self._resolve_default_bc("placement") == BCType.DIRICHLET
        last_dirichlet = max((i for i, seg in enumerate(order) if seg.bc_type == BCType.DIRICHLET), default=-1)
        return [
            seg
            for i, seg in enumerate(order)
            if (
                seg.region is not None
                or seg.sdf_region is not None
                or (
                    seg.region_name is not None
                    and (parse_boundary_face(seg.region_name) is None or _marks_region(geometry, seg.region_name))
                )
            )
            and (seg.bc_type == BCType.DIRICHLET or i < last_dirichlet or dirichlet_default)
        ]

    def segment_for_corner(
        self,
        faces: list[BoundaryFace],
        geometry: Any = None,
        equivalent: Callable[[BCSegment, BCSegment], bool] | None = None,
    ) -> BCSegment:
        """The segment that governs a node shared by several faces -- a corner, or an edge in 3-D (#2512 B1).

        Each face's own segment comes from :meth:`segment_for_face`; ``corner_strategy`` then chooses among
        them (:meth:`_corner_choice`). A point at the corner is resolved at the point (:meth:`segment_for_point`).
        """
        return self._corner_resolution(tuple(faces), geometry, equivalent)[1]

    def corner_nodes(
        self,
        shape: tuple[int, ...],
        geometry: Any = None,
        equivalent: Callable[[BCSegment, BCSegment], bool] | None = None,
    ) -> dict[tuple[int, ...], tuple[BoundaryFace, BCSegment]]:
        """For a reader that writes conditions face by face: every node of a ``shape`` grid that several faces share,
        with the segment that governs it and the face whose segment it is (:meth:`segment_for_corner`, asked with the
        reader's own ``equivalent``). The reader writes its faces, then writes each node's segment, along that face,
        at the node, so the node takes the owner's choice whatever order the faces were written in (#2512 row B1).
        Under ``corner_strategy="average"`` or ``"mollify"`` a node whose faces' segments impose different conditions
        is refused here."""
        geometry = self._query_geometry(geometry)
        key = ("corner nodes", tuple(shape), equivalent)
        return self._owner_memo(
            key,
            geometry,
            lambda: {
                node: self._corner_resolution(faces, geometry, equivalent)
                for node, faces in _multi_face_nodes(tuple(shape))
            },
        )

    def _corner_resolution(
        self, faces: tuple[BoundaryFace, ...], geometry: Any, equivalent: Any
    ) -> tuple[BoundaryFace, BCSegment]:
        """The face and the segment a node shared by ``faces`` takes: :meth:`segment_for_corner`'s rule."""
        governing = [self.segment_for_face(face, geometry, equivalent) for face in faces]
        chosen = self._corner_choice(governing, ", ".join(f.to_string() for f in faces), equivalent)
        return faces[next(i for i, seg in enumerate(governing) if seg is chosen)], chosen

    def _corner_choice(
        self, governing: list[BCSegment], where: str, equivalent: Callable[[BCSegment, BCSegment], bool] | None = None
    ) -> BCSegment:
        """Among the segments governing a corner's faces, the one the corner takes: the first in the corner order
        under ``corner_strategy="priority"``. ``"average"`` and ``"mollify"`` blend outward normals and say nothing
        about which condition applies, so where the segments impose different conditions they are refused; where they
        impose one, there is nothing to choose."""
        same = _same_condition if equivalent is None else equivalent
        first = min(governing, key=self._corner_key)
        if self.corner_strategy != "priority" and not all(same(seg, first) for seg in governing):
            raise ValueError(
                f"BoundaryConditions: the corner shared by {where} is governed by "
                f"{', '.join(repr(seg.name) for seg in governing)}, which impose different conditions, and "
                f"corner_strategy={self.corner_strategy!r} blends outward normals -- it does not say which "
                "condition applies there. Use corner_strategy='priority' (#2512 row B1)."
            )
        return first

    def segment_for_point(
        self,
        point: np.ndarray,
        face: BoundaryFace | str | None = None,
        geometry: Any = None,
        tolerance: float = SDF_BOUNDARY_TOL,
        axis_names: dict[int, str] | None = None,
        equivalent: Callable[[BCSegment, BCSegment], bool] | None = None,
    ) -> BCSegment:
        """The segment that governs the boundary point ``point`` (#2512 row B1).

        The point-level reading of the same placement rule as :meth:`segment_for_face`, each segment
        tested by :meth:`BCSegment.matches_point`: on a face that one governs whole, both answer the
        same. The faces a point lies on are read from ``domain_bounds``, or, when the BC carries none, from
        the bounds of the geometry the query is decided on (``geometry``, else the grid this BC is bound to).

        **A point at a corner is resolved at the point** (user ruling, 2026-10-14). Without ``face``, a point
        on several faces takes each face's own segment, read at the point only where a partial cover keeps the face
        from answering (:meth:`_face_resolution`): a ``region`` or ``sdf_region`` covering part of it ahead of the
        segment governing the rest with another condition, or covering part of a face no segment covers whole and
        whose ``default_bc`` does not answer. Then the segment holding the point on that face (this method with
        ``face``) stands in. Every other refusal stands; a segment shadowed on every face refuses every corner wherever
        the shadow check can enumerate the faces -- the BC's ``dimension`` is set, or the geometry is a grid.
        ``corner_strategy`` then chooses among them in the corner order (:meth:`_corner_choice`). Both steps are
        asked with ``equivalent``, so wherever the node asked with the same ``equivalent`` answers, the point gives
        the node's segment; and where the node refuses only because a face is partly covered, the point answers.
        With ``face``, the point is read on that face alone and the corner rule does not apply.

        **A point read on a face is read at its nearest point of the box** (#2780), by every placement field. A particle
        past a wall, read on the face it crossed, takes the segment holding that nearest point, which stands in for the
        point where it crossed until #2733 passes that point. Before, only a marked ``region_name`` read it there, and a
        ``region`` or ``sdf_region`` exit holding only the face's own coordinate never held a stepped point, so the
        particles crossed it under ``default_bc``. A point on no face is read where it is.

        Args:
            point: Spatial coordinates, shape (dimension,).
            face: The face the caller knows the point is on. Without it, every face the point lies on
                is read from the bounds, or the boundary is identified from ``domain_sdf``.
            geometry: The grid whose marked-region mask places a ``region_name`` segment (a name that is a face
                label is read without one), and the source of the bounds when the BC carries none.
            tolerance: How far off a face a point may lie and still be on it.
            equivalent: The consumer's sameness of two segments, as :meth:`segment_for_face` and
                :meth:`segment_for_corner` take it; by default `_same_condition`. At a corner it decides whether a
                face answers and which segment the corner takes, so the point and the node asked with the same
                ``equivalent`` agree.
        """
        point = np.asarray(point, dtype=float)
        geometry = self._query_geometry(geometry)
        if face is None:
            on, bounds = self._point_faces(point, geometry, tolerance, "segment_for_point")
            if len(on) > 1:
                at_point = [
                    seg for _, seg in self._corner_readings(point, on, geometry, tolerance, axis_names, equivalent)
                ]
                return self._corner_choice(at_point, ", ".join(f.to_string() for f in on), equivalent)
            faces: list[str | None] = [f.to_string() for f in on] or [None]
        else:
            bounds = self._query_bounds(geometry)
            faces = [face.to_string() if isinstance(face, BoundaryFace) else face]
        on_box = point
        if bounds is not None and point.shape == (bounds.shape[0],):
            on_box = np.clip(point, bounds[:, 0], bounds[:, 1])
        governing = []
        for boundary_id in faces:
            for segment in self._face_order():
                if segment.matches_point(
                    point if boundary_id is None else on_box,
                    boundary_id,
                    bounds,
                    tolerance,
                    axis_names,
                    domain_sdf=self.domain_sdf,
                    geometry=geometry,
                ):
                    governing.append(segment)
                    break
            else:
                governing.append(self._default_segment("segment_for_point", f"the boundary point {point.tolist()}"))
        return governing[0]

    def face_segments_at_point(
        self,
        point: np.ndarray,
        geometry: Any = None,
        tolerance: float = SDF_BOUNDARY_TOL,
        axis_names: dict[int, str] | None = None,
        equivalent: Callable[[BCSegment, BCSegment], bool] | None = None,
    ) -> list[tuple[BoundaryFace, BCSegment, np.ndarray]]:
        """Every face ``point`` lies on (:meth:`_point_faces`), in face order, with the segment that face takes there
        and the outward normal a row along it writes: the readings :meth:`segment_for_point`, asked with the same
        arguments, chooses among (#2512 row B1). On one face, the point's segment on it; at a corner, each face's own
        answer, read at the point only where a partial cover keeps the face from answering
        (:meth:`_corner_face_at_point`). A reader that writes a row along a face's normal picks that face among these.
        Empty when the point lies on no face.

        The normal is the face's (:meth:`outward_normal_for_face`) where the faces are read from a box. Where they are
        read from ``domain_sdf``, the one face is the gradient's dominant axis, which on a curved boundary is not its
        normal, so the normal is the SDF's at the point (:meth:`get_outward_normal`), the function the uniform path's
        SDF branch calls. The two paths agree wherever the uniform path reaches that branch, which is off the
        geometry's box: at a kink of ``domain_sdf`` lying on the geometry's box, the uniform path writes the box face's
        normal and this one the SDF's (#2675)."""
        point = np.asarray(point, dtype=float)
        geometry = self._query_geometry(geometry)
        on, bounds = self._point_faces(point, geometry, tolerance, "face_segments_at_point")
        if len(on) == 1:
            readings = [(on[0], self.segment_for_point(point, on[0], geometry, tolerance, axis_names))]
        else:
            readings = self._corner_readings(point, on, geometry, tolerance, axis_names, equivalent)
        if bounds is None and readings:
            normal = self.get_outward_normal(point)
            if normal is None:
                raise ValueError(
                    "BoundaryConditions.face_segments_at_point: the gradient of domain_sdf vanishes at "
                    f"{point.tolist()}, so it gives this boundary point no outward normal (#2512 row B1)."
                )
            return [(face, segment, normal) for face, segment in readings]
        return [
            (face, segment, self.outward_normal_for_face(face, dimension=point.shape[0])) for face, segment in readings
        ]

    def _point_faces(
        self, point: np.ndarray, geometry: Any, tolerance: float, reader: str
    ) -> tuple[list[BoundaryFace], np.ndarray | None]:
        """The faces ``point`` lies on, for both point readers (#2512 row B1), and the box they were read from: every
        face of the box (:meth:`_query_bounds`) within ``tolerance``; else, for a BC with ``domain_sdf``, the one face
        its SDF gives the point (:meth:`identify_boundary_face`, the gradient's dominant axis), or none where the SDF
        puts the point off its boundary; else refused. Under an SDF a point has one face, so no corner rule applies
        there (#2675)."""
        bounds = self._query_bounds(geometry)
        if bounds is not None:
            return self._faces_at_point(point, tolerance, bounds), bounds
        if self.domain_sdf is not None:
            face = self.identify_boundary_face(point, tolerance)
            return ([] if face is None else [face]), None
        raise ValueError(
            f"BoundaryConditions.{reader}: this BC has no domain_bounds or domain_sdf, and the query's "
            f"geometry ({type(geometry).__name__}) gives no box to read the faces from: it is not a "
            "GeometryProtocol whose get_bounds() returns bounds. Pass such a geometry, or set domain_bounds "
            "(#2512 row B1)."
        )

    def _corner_readings(
        self,
        point: np.ndarray,
        faces: list[BoundaryFace],
        geometry: Any,
        tolerance: float,
        axis_names: dict[int, str] | None,
        equivalent: Callable[[BCSegment, BCSegment], bool] | None,
    ) -> list[tuple[BoundaryFace, BCSegment]]:
        """Each of a corner's ``faces`` with its segment at ``point`` (:meth:`_corner_face_at_point`)."""
        return [(f, self._corner_face_at_point(point, f, geometry, tolerance, axis_names, equivalent)) for f in faces]

    def segments_for_facets(self, facets: BoundaryFacets) -> list[BCSegment]:
        """The segment that governs each boundary facet of a mesh, one per row of ``facets`` (#2512 row B1, stage B1b).

        **The face owner's order and sameness rule**, :meth:`_face_order` and `_same_condition`, applied per facet: the
        first segment in that order that holds a facet governs it, and a facet no segment holds takes ``default_bc``
        (:meth:`_default_segment`, refused when unset or ROBIN). A segment holds a facet when every placement field it
        carries does:

        - ``boundary``: a name that is a tag of the mesh is that tag, so a mesh's own group called ``"left"`` is that
          group. Only a name that is no tag is read as the box face it aliases (``parse_boundary_face``), and holds the
          facets of the mesh's tag for that face (``"left"`` -> ``x_min``). A name that is neither is refused, naming
          both readings. ``"all"`` holds every facet.
        - ``region_name``: the tag of that name.
        - ``region``, ``sdf_region``: read at the facet's midpoint, as the point owner reads them at a point
          (`region_holds`, `sdf_region_holds`), so a region that covers some of a face's facets holds those facets.
          A facet is this reader's unit, which is why the face owner's partial-cover refusal does not apply here.
        - ``normal_direction``: read at the facet's outward normal (`normal_holds`).

        A segment that governs no facet is refused, as on a box (:meth:`_check_shadowed`), unless it imposes the same
        condition as whatever governs the facets it holds, or every facet when it holds none. A node that facets
        governed by different segments share takes :meth:`segment_for_shared_node`.
        """
        order = self._face_order()
        n_facets = facets.midpoints.shape[0]
        held = [self._facet_cover(segment, facets) for segment in order]
        winner = np.full(n_facets, -1)
        for k, mask in enumerate(held):
            winner[(winner < 0) & mask] = k
        segments = list(order)
        if (winner < 0).any():
            unheld = int((winner < 0).sum())
            segments.append(
                self._default_segment(
                    "BoundaryConditions.segments_for_facets", f"{unheld} boundary facet(s) of the mesh"
                )
            )
            winner[winner < 0] = len(order)
        governing = [int(k) for k in np.unique(winner)]
        governs = {id(segments[k]) for k in governing}
        touched = {
            id(segment): [
                (f"{int((held[k] & (winner == j)).sum())} facet(s)", segments[j]) for j in np.unique(winner[held[k]])
            ]
            for k, segment in enumerate(order)
            if id(segment) not in governs
        }
        nowhere, dead = _segments_governing_nothing(
            order, governs, touched, set(), [segments[k] for k in governing], _same_condition
        )
        if nowhere:
            holding = sorted(tag for tag, rows in facets.tags.items() if len(rows))
            raise ValueError(
                "BoundaryConditions: "
                + "; ".join(f"segment {seg.name!r} {self._why_no_facet(seg, facets)}" for seg in nowhere)
                + ". A segment that governs no facet would have its condition dropped where it differs from what "
                f"governs there. The mesh's tags that hold facets: {holding}. Tag the facets the condition is meant "
                "for and name that tag, or remove the segment (#2512 row B1, #2595)."
            )
        if dead:
            where = " | ".join(
                f"{seg.name!r} (priority {seg.priority}) -- "
                + "; ".join(f"{at}: {win.name!r} (priority {win.priority}) outranks it" for at, win in hits)
                for seg, hits in dead
            )
            raise ValueError(
                f"BoundaryConditions: these segments govern no facet of the mesh, so their conditions would be "
                f"dropped: {where}. Order on a facet: priority, then a placed segment before a global one (no "
                "placement, or boundary='all'), then declaration order. Remove it or the segment that outranks it, or "
                "place the two on different facets (#2512 row B1)."
            )
        return [segments[k] for k in winner]

    def segment_for_shared_node(self, governing: list[BCSegment], where: str) -> BCSegment:
        """The segment a node takes when facets governed by different segments share it (#2512 row B1, stage B1b).

        A grid node on several faces takes the corner rule (:meth:`segment_for_corner`); a mesh node on facets
        governed by different segments takes the same rule among those segments (:meth:`_corner_choice`), so a
        Dirichlet facet meeting a natural one at equal priority holds the node, and ``corner_strategy="average"`` or
        ``"mollify"`` refuses a node whose segments impose different conditions."""
        return self._corner_choice(governing, where)

    def refuse_a_segment_left_no_node(self, segment: BCSegment, takers: list[BCSegment], where: str) -> None:
        """(c) at a reader's own unit, the node (#2512 row B1, stage B1b). ``takers`` are the segments that
        :meth:`segment_for_shared_node` gave the nodes of a DIRICHLET segment that governs facets but holds none of their
        nodes -- a single facet between two higher-priority segments is one. If any of them imposes the same condition
        (`_same_condition`), that condition still holds at a node of its facets; if none does, its condition is
        imposed nowhere, and it is refused."""
        if any(_same_condition(segment, taker) for taker in takers):
            return
        names = ", ".join(sorted({repr(t.name) for t in takers}))
        raise ValueError(
            f"BoundaryConditions: segment {segment.name!r} governs {where} but holds none of their nodes: the corner rule "
            f"gives each to {names}, {'which imposes' if len(takers) == 1 else 'each imposing'} a different condition, "
            "so its condition would be dropped. Place it on only the facets it should govern (with a region, for "
            f"instance) and give it a higher priority than {names}, so that it takes the nodes it shares with them; or "
            "remove it (#2512 row B1)."
        )

    def _facet_cover(self, segment: BCSegment, facets: BoundaryFacets) -> np.ndarray:
        """Which rows of ``facets`` ``segment`` holds: every placement field it carries, read per facet."""
        held = np.ones(facets.midpoints.shape[0], dtype=bool)
        if segment.boundary is not None and segment.boundary != "all":
            held &= self._facets_named(segment, "boundary", segment.boundary, facets)
        if segment.region_name is not None:
            held &= self._facets_named(segment, "region_name", segment.region_name, facets)
        if segment.normal_direction is not None:
            held &= normal_holds(segment, facets.normals, "a mesh facet's outward normal")
        if segment.region is not None:
            held &= region_holds(segment, facets.midpoints, SDF_BOUNDARY_TOL)
        if segment.sdf_region is not None:
            held &= sdf_region_holds(segment, facets.midpoints, SDF_BOUNDARY_TOL)
        return held

    def _why_no_facet(self, segment: BCSegment, facets: BoundaryFacets) -> str:
        """Why ``segment`` holds no facet, for the refusal in :meth:`segments_for_facets`."""
        for field_name in ("boundary", "region_name"):
            name = getattr(segment, field_name)
            if name is not None and name != "all":
                tag = self._tag_named(segment, field_name, name, facets)
                if not len(facets.tags[tag]):
                    return f"names the tag {tag!r} ({field_name}={name!r}), which holds no facet"
        return "holds no facet: no facet midpoint is in its region or sdf_region, or no facet normal is its normal_direction"

    @classmethod
    def _facets_named(cls, segment: BCSegment, field_name: str, name: Any, facets: BoundaryFacets) -> np.ndarray:
        """The rows of the tag ``name`` reads as (:meth:`_tag_named`)."""
        rows = np.zeros(facets.midpoints.shape[0], dtype=bool)
        rows[np.asarray(facets.tags[cls._tag_named(segment, field_name, name, facets)], dtype=np.intp)] = True
        return rows

    @staticmethod
    def _tag_named(segment: BCSegment, field_name: str, name: Any, facets: BoundaryFacets) -> str:
        """The tag ``name`` reads as: the tag of that name, else (``boundary`` only) the tag of the box face it
        aliases. Refused when it is neither, or when the tag holds facets inside the domain."""
        tag = name if isinstance(name, str) and name in facets.tags else None
        alias = parse_boundary_face(name) if field_name == "boundary" and tag is None else None
        if tag is None and alias is not None and alias.to_string() in facets.tags:
            tag = alias.to_string()
        if tag is None:
            as_face = (
                f"as the box face it aliases, {alias.to_string()!r}, which the mesh does not tag"
                if alias is not None
                else "as a box face, which it names none of"
                if field_name == "boundary"
                else "as nothing else: a region_name is read only as a tag"
            )
            holding = sorted(t for t, rows in facets.tags.items() if len(rows))
            tags_held = (
                f"the mesh's tags: {sorted(facets.tags)}, of which {holding} hold facets"
                if facets.tags
                else "the mesh has no tags"
            )
            raise ValueError(
                f"BC segment {segment.name!r} names {field_name}={name!r}, which this mesh does not resolve. Read as a "
                f"tag of the mesh, it is none ({tags_held}); read {as_face}. Name one of the mesh's tags (#2512 row B1)."
            )
        if tag in facets.off_boundary:
            raise ValueError(
                f"BC segment {segment.name!r} names {field_name}={name!r}, the mesh tag {tag!r}, which also holds "
                "facets inside the domain. A boundary condition governs boundary facets only, so a condition on an "
                "interior facet would be dropped (#2512 row B1)."
            )
        return tag

    def _face_order(self) -> list[BCSegment]:
        """Segments in the owner's order: priority descending, placed before global, then declaration order.

        ``self.segments`` is already priority-descending and stable on declaration order (``__post_init__``),
        so a stable sort on the specificity level alone gives the full order.
        """
        return sorted(self.segments, key=lambda seg: (-seg.priority, _is_global(seg)))

    def _corner_key(self, segment: BCSegment) -> tuple:
        """Precedence at a corner: priority, then essential (DIRICHLET) before natural, then placed before global,
        then declaration order. A synthesised ``default_bc`` segment is global and declared last."""
        position = next((i for i, seg in enumerate(self.segments) if seg is segment), len(self.segments))
        return (-segment.priority, segment.bc_type != BCType.DIRICHLET, _is_global(segment), position)

    def _owner_memo(self, key: tuple, geometry: Any, compute: Callable[[], Any]) -> Any:
        """The owner's per-object memo -- covers and the shadow check -- one entry per ``key`` and ``geometry``.

        The geometry is keyed by ``id()``, which CPython gives a new object once the old one is freed: a cover read
        on a 3 x 3 grid would then answer for an 11 x 11 grid built at the same address, where the same segment
        covers only part of the face. So an entry keeps a weak reference to its geometry and is used only while
        that same object is alive at the key's id -- the precedent of `types.callable_protocols` (#2406). A
        geometry that cannot be weakly referenced is not memoised. Not a dataclass field, so it takes no part in
        equality, and ``dataclasses.replace`` starts a fresh one. A BC mutated after its first resolve is not
        re-checked.
        """
        cache = self.__dict__.setdefault("_b1_owner_cache", {})
        slot = (*key, id(geometry))
        entry = cache.get(slot)
        if entry is not None and (entry[0] is None if geometry is None else entry[0]() is geometry):
            return entry[1]
        value = compute()
        if geometry is None:
            cache[slot] = (None, value)
        elif type(geometry).__weakrefoffset__:  # the type supports weak references
            cache[slot] = (weakref.ref(geometry), value)
        return value

    def _cached_face_cover(self, segment: BCSegment, face: BoundaryFace, geometry: Any) -> str:
        return self._owner_memo(
            ("cover", id(segment), face), geometry, lambda: self._face_cover(segment, face, geometry)
        )

    def _refuse_shadowed_segments(self, face: BoundaryFace, geometry: Any, equivalent: Any) -> None:
        """Refuse, once per (dimension, equivalence), a segment that governs no face of the domain (#2512 B1).

        A segment no face resolves to is dropped silently -- by a priority typo, a misplaced catch-all, a
        misnamed face -- unless it imposes the same condition as the segment that outranks it on every face
        it touches (a duplicate, an effect-neutral strip), which changes nothing. The domain's faces come
        from ``self.dimension``, or the geometry's; with neither, the check cannot enumerate faces and is
        skipped. Cached per object: a BC mutated after its first resolve is not re-checked.
        """
        from mfgarchon.geometry.base import CartesianGrid  # local: geometry.base imports this package

        dimension = self.dimension
        if dimension is None and isinstance(geometry, CartesianGrid):
            dimension = len(geometry.get_grid_shape())
            self._refuse_missing_axes(dimension)  # an unbound BC: the geometry is where the dimension comes from
        if dimension is None:
            return
        if equivalent is None:
            self._owner_memo(("shadow", dimension), geometry, lambda: self._check_shadowed(dimension, geometry, None))
        else:
            self._check_shadowed(dimension, geometry, equivalent)

    def _check_shadowed(self, dimension: int, geometry: Any, equivalent: Any) -> bool:
        """The shadow check itself, for :meth:`_refuse_shadowed_segments`: raises, or returns True.

        A segment governs a face, or it is shadowed where it reaches, or it covers part of a face ahead of that face's
        winner (the partial-cover rule's case). A segment that does none of these reaches no face: it is read as
        shadowed on every face, so the reader's own ``equivalent`` decides, as for any shadowed segment -- one
        equivalent to every face's segment changes nothing there and passes, and any other is refused (#2634,
        review 2: a REFLECTING strip between two grid nodes changes no axis's periodicity, so the particle topology
        query accepts it). A node several faces share takes the corner choice among those faces' segments
        (:meth:`segment_for_corner`), so a placement holding only such nodes reaches no face either.
        """
        same = _same_condition if equivalent is None else equivalent
        faces = [BoundaryFace(axis, side) for axis in range(dimension) for side in ("min", "max")]
        order = self._face_order()
        touched: dict[int, list[tuple[str, BCSegment]]] = {id(seg): [] for seg in order}
        governs: set[int] = set()
        partial_ahead: set[int] = set()  # covers part of a face ahead of its winner: the partial-cover rule's case
        face_winners: list[BCSegment] = []
        for f in faces:
            # A segment ahead of the face's winner covers only part of it: that is the partial-cover rule's
            # case, refused or accepted when the face is resolved. A segment after the winner is shadowed there.
            winner = None
            shadowed = []
            for seg in order:
                cover = self._cached_face_cover(seg, f, geometry)
                if cover == "none":
                    continue
                if winner is not None:
                    shadowed.append(seg)
                elif cover == "whole":
                    winner = seg
                else:
                    partial_ahead.add(id(seg))
            if winner is None:
                if self._default_refusal("segment_for_face", f"the {f.to_string()} face") is not None:
                    continue  # refused when this face itself is resolved, with its own message
                winner = self._default_segment("segment_for_face", f"the {f.to_string()} face")
            else:
                governs.add(id(winner))
            face_winners.append(winner)
            for seg in shadowed:
                touched[id(seg)].append((f.to_string(), winner))
        nowhere_segments, dead_segments = _segments_governing_nothing(
            order, governs, touched, partial_ahead, face_winners, same
        )
        nowhere = [seg.name for seg in nowhere_segments]
        dead = [
            f"{seg.name!r} (priority {seg.priority}) -- "
            + "; ".join(f"{at}: {winner.name!r} (priority {winner.priority}) outranks it" for at, winner in hits)
            for seg, hits in dead_segments
        ]
        if nowhere:
            raise ValueError(
                f"BoundaryConditions: segment(s) {', '.join(repr(name) for name in nowhere)} cover no node of any "
                f"face's interior{' on this grid' if geometry is not None else ''}, so their conditions would be "
                "dropped where they differ from what governs there. Their placement selects no node inside a face "
                "-- a region or sdf_region between the boundary nodes or off the face it names, a normal_direction "
                "no face points along, a marked region on no face -- or only nodes that faces share, and such a "
                "node takes the corner choice among those faces' segments. Place them on a face, or remove them "
                "(#2512 row B1)."
            )
        if dead:
            raise ValueError(
                "BoundaryConditions: these segments govern no face of the domain, so their conditions would be "
                "dropped: " + " | ".join(dead) + ". Order on a face: priority, then a placed segment before a "
                "global one (no placement, or boundary='all'), then declaration order. Raise the segment's "
                "priority, or remove it (#2512 row B1)."
            )
        return True

    def _corner_face_at_point(
        self,
        point: np.ndarray,
        face: BoundaryFace,
        geometry: Any,
        tolerance: float,
        axis_names: dict[int, str] | None,
        equivalent: Callable[[BCSegment, BCSegment], bool] | None,
    ) -> BCSegment:
        """One face's segment for a point at a corner (:meth:`segment_for_point`): the face's own answer
        (:meth:`_face_resolution`) wherever it has one, so the point and the node asked with the same ``equivalent``
        agree; where a partial cover keeps the face from answering (:meth:`_face_resolution`'s second and third
        outcomes), the segment holding the point on that face.
        Every other refusal stands. Reading every face at the point instead would let a segment holding the point at
        a higher priority override the face's answer -- review 2's B-c, a strip accepted on x_max as the wall's
        condition, would take the y_max Dirichlet's corner."""
        governing, part = self._face_resolution(face, geometry, equivalent, "BoundaryConditions.segment_for_point")
        if part is None:
            return governing
        return self.segment_for_point(point, face, geometry, tolerance, axis_names)

    def _query_bounds(self, geometry: Any) -> np.ndarray | None:
        """The box a query reads faces from, shape (dimension, 2): ``domain_bounds``; else, for a BC with no
        ``domain_sdf`` either, the bounds of ``geometry`` when it has them; else None. The point owner reads a point's faces from it, and the face owner a ``region``'s
        cover when there is no grid. A solver's BC seldom carries bounds of its own; GFDM's never does, and before the
        point readers had one owner it passed its geometry's bounds to them itself (#2512 row B1)."""
        if self.domain_bounds is not None:
            return np.asarray(self.domain_bounds, dtype=float)
        if self.domain_sdf is not None:
            return None  # the BC's own domain decides; a geometry's bounding box would override it
        from mfgarchon.geometry.protocol import GeometryProtocol  # local: the geometry package imports this one

        if not isinstance(geometry, GeometryProtocol):
            return None
        box = geometry.get_bounds()
        if box is None:
            return None
        return np.column_stack([np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float)])

    def _faces_at_point(
        self, point: np.ndarray, tolerance: float, bounds: np.ndarray | None = None
    ) -> list[BoundaryFace]:
        """Every box face ``point`` lies on: one on a face's interior, two or more at a corner. ``bounds``
        defaults to ``domain_bounds``."""
        bounds = np.asarray(self.domain_bounds if bounds is None else bounds, dtype=float)
        faces = []
        for axis in range(bounds.shape[0]):
            for side, bound in (("min", bounds[axis, 0]), ("max", bounds[axis, 1])):
                if abs(point[axis] - bound) <= tolerance + abs(bound) * BOUNDARY_REL_TOL:
                    faces.append(BoundaryFace(axis, side))
        return faces

    def _face_cover(self, segment: BCSegment, face: BoundaryFace, geometry: Any) -> str:
        """``"whole"``, ``"partial"`` or ``"none"``: how much of ``face`` ``segment`` is placed on."""
        if segment.boundary is not None and segment.boundary != "all":
            named = parse_boundary_face(segment.boundary)
            if named is None:
                raise ValueError(
                    f"BC segment {segment.name!r} is placed on boundary {segment.boundary!r}, which names no "
                    f"face of a box domain, so a face-level reader cannot tell whether it governs "
                    f"{face.to_string()}. Name a face ('x_min', 'left', 'axis4_max', ...) or 'all' "
                    "(#2512 row B1)."
                )
            if named != face:
                return "none"
        if segment.region_name is not None:
            return "whole" if region_name_governs_face(segment, face, geometry) else "none"
        # Each placement field narrows the cover: a normal picks the face, and an sdf_region beside it (BCSegment allows
        # the pair) still decides how much of that face the segment holds.
        if segment.normal_direction is not None and not _normal_selects_face(segment, face):
            return "none"
        if segment.region is not None or segment.sdf_region is not None:
            return self._sub_face_cover(segment, face, geometry)
        return "whole"

    def _sub_face_cover(self, segment: BCSegment, face: BoundaryFace, geometry: Any) -> str:
        """``"whole"``, ``"partial"`` or ``"none"``: how much of ``face`` a region/sdf placement holds."""
        mask = _placement_mask_on_grid(segment, geometry)
        if mask is not None:
            covered, partial = _region_faces(mask)
            return "whole" if face in covered else "partial" if face in partial else "none"
        if segment.sdf_region is not None:
            raise ValueError(
                f"BC segment {segment.name!r} is placed by sdf_region; which faces it covers is decided on "
                f"a grid's boundary nodes, and this reader has no grid geometry "
                f"({type(geometry).__name__}). Pass the grid, or place the segment with boundary= "
                "(#2512 row B1)."
            )
        bounds = self._query_bounds(geometry)
        if bounds is None:
            remedy = (
                "this BC's domain is its domain_sdf, so a geometry's bounding box is not read for it: pass the grid or "
                "set domain_bounds"
                if self.domain_sdf is not None
                else "pass a grid or a geometry with bounds, or set domain_bounds"
            )
            raise ValueError(
                f"BC segment {segment.name!r} is placed by region, and neither a grid geometry nor a box (domain_bounds, "
                "or the bounds of the geometry for a BC with no domain_sdf) says how far each face extends, so this "
                f"reader cannot tell whether it covers {face.to_string()}: {remedy} (#2512 row B1)."
            )
        tol = SDF_BOUNDARY_TOL
        part_of_face = False
        for key, (low, high) in segment.region.items():
            axis = _region_axis(segment, key, bounds.shape[0])
            b_low, b_high = bounds[axis]
            if axis == face.axis:
                coordinate = b_low if face.side == "min" else b_high
                if coordinate < low - tol or coordinate > high + tol:
                    return "none"
            elif high <= b_low + tol or low >= b_high - tol:
                return "none"
            elif not (low <= b_low + tol and high >= b_high - tol):
                part_of_face = True
        return "partial" if part_of_face else "whole"

    def _segment_for_tag(self, tag: Any) -> BCSegment:
        """A name that is no box face (a mesh tag): governed by equal name, or by an unplaced segment."""
        for segment in self.segments:
            if segment.boundary == tag or segment.region_name == tag:
                return segment
            if segment.boundary is None and segment.region_name is None:
                if segment.region is not None or segment.sdf_region is not None or segment.normal_direction is not None:
                    raise ValueError(
                        f"BC segment {segment.name!r} is placed by region, sdf_region or normal_direction, "
                        f"which cannot be resolved against the named boundary {tag!r} (#2512 row B1)."
                    )
                return segment
        return self._default_segment("segment_for_face", f"the boundary named {tag!r}")

    def _default_segment(self, context: str, where: str = "this location") -> BCSegment:
        """The condition on a location no segment governs: ``default_bc`` with ``default_value``, refused where
        :meth:`_default_refusal` says why."""
        refusal = self._default_refusal(context, where)
        if refusal is not None:
            raise ValueError(refusal)
        return BCSegment(
            name="default", bc_type=self._resolve_default_bc(context), value=self.default_value, priority=-1
        )

    def _default_refusal(self, context: str, where: str) -> str | None:
        """Why ``default_bc`` cannot govern ``where``, or None when it can: the one predicate :meth:`_default_segment`
        raises on and :meth:`_face_resolution` reads.

        Refused when ``default_bc`` is unset (#1100) -- the remedy is to set it or to cover every face
        -- and when it is ROBIN: ``default_bc`` is a bare ``BCType``, and this class carries no default
        alpha/beta, so the segment would take ``BCSegment``'s ``alpha=1.0, beta=0.0``, which is the
        Dirichlet corner of the Robin family (#2042).
        """
        if self.default_bc is None:
            return (
                f"{context}: no segment governs {where}, and default_bc was not specified on BoundaryConditions "
                f"(#1100). Either cover {where} with a segment -- an unplaced segment says 'the rest' -- or pass "
                "default_bc= with the condition every location no segment governs should get "
                "(#2512 row B1)."
            )
        if self._resolve_default_bc(context) == BCType.ROBIN:
            return (
                f"{context}: default_bc is ROBIN, but default_bc is a bare BCType -- it carries no alpha and no "
                "beta, and this class has no default alpha/beta -- so a location no segment governs would get "
                "BCSegment's alpha=1.0, beta=0.0, which is a DIRICHLET wall, not a Robin one. Give every face an "
                "explicit BCSegment carrying its own alpha and beta (an unplaced segment says 'the rest'), or "
                "choose a default_bc whose condition needs no coefficients (#2042)."
            )
        return None

    def get_bc_at_point(
        self,
        point: np.ndarray,
        boundary_id: str | None = None,
        tolerance: float = SDF_BOUNDARY_TOL,
        axis_names: dict[int, str] | None = None,
        geometry=None,  # Type: SupportsRegionMarking | None (Issue #596 Phase 2.5)
    ) -> BCSegment:
        """The segment that applies at a boundary point: :meth:`segment_for_point` (#2512 row B1)."""
        return self.segment_for_point(point, boundary_id, geometry, tolerance, axis_names)

    def get_bc_type_at_boundary(self, boundary: BoundaryFace | str) -> BCType:
        """The BC type that governs a face: :meth:`segment_for_face`'s segment (#2512 row B1).

        Examples:
            >>> bc = neumann_bc(dimension=2)
            >>> bc.get_bc_type_at_boundary("x_min")
            BCType.NEUMANN
        """
        return self.segment_for_face(boundary).bc_type

    def get_bc_value_at_boundary(
        self,
        boundary: BoundaryFace | str,
        time: float | None = None,
        point: np.ndarray | None = None,
        geometry: Any = None,
    ) -> float:
        """The BC value on a face: :meth:`segment_for_face`'s segment, read by its one owner.

        Args:
            boundary: Boundary identifier (e.g., "x_min", "x_max")
            time: The time to read a time-dependent value at. With none, a value that declares ``t`` is refused,
                not read at a default time (#2512 row B2, #2535); a constant, or ``g(x)``, needs none.
            point: Optional spatial point for spatially-varying BCs
            geometry: The solver's grid, which decides a region or sdf_region placement (:meth:`segment_for_face`).
        """
        where = f"BoundaryConditions.get_bc_value_at_boundary({boundary!r})"
        segment = self.segment_for_face(boundary, geometry)
        return segment_value(segment, t=time, x=point, where=where, time_keyword="time")

    def identify_boundary_face(
        self,
        point: np.ndarray,
        tolerance: float = BOUNDARY_TOL,
        domain_bounds: np.ndarray | None = None,
    ) -> BoundaryFace | None:
        """
        Identify which boundary face a point lies on (dimension-agnostic).

        Returns a BoundaryFace(axis, side) for rectangular domains,
        or normal-based face for SDF domains.

        Args:
            point: Spatial coordinates.
            tolerance: Closed-inequality tolerance for boundary detection
                (``|point[axis] - bound| <= tolerance``). Default 1e-6 covers
                collocation generators that place boundary points at ε=1e-6
                off the wall to avoid SDF coincidence. Adjust larger if your
                collocation ε is larger.
            domain_bounds: Optional override for axis-aligned bounds. If
                supplied, takes precedence over ``self.domain_bounds`` for
                this call only. Useful when a solver knows the geometry
                bounds but the BC spec doesn't carry them.

        Returns:
            BoundaryFace or None if not on boundary.

        Note:
            Uses ``<=`` (closed inequality) rather than ``<`` (strict). With
            strict ``<``, a point at exactly ``tolerance`` distance from the
            wall would fall through to None — and floating-point rounding
            decides whether two symmetric walls classify identically.
        """
        dimension = self._require_dimension("identify_boundary_face")
        point = np.asarray(point, dtype=float)

        # Prefer caller-supplied bounds when provided
        bounds = domain_bounds if domain_bounds is not None else self.domain_bounds

        # Method 1: Rectangular domain (axis-aligned detection).
        # Use hybrid absolute+relative tolerance to absorb floating-point
        # arithmetic noise at non-trivial bound magnitudes. Example: at
        # bound=20, `(20.0 - 1e-6) - 20.0` computes to ~1.0000000003e-6
        # (3e-14 above the mathematical 1e-6) purely from FP subtraction
        # error. A small relative term (1e-12 * |bound|) absorbs this while
        # remaining far below any realistic interior-point distance.
        if bounds is not None:
            bounds = np.asarray(bounds, dtype=float)
            for axis_idx in range(dimension):
                low, high = bounds[axis_idx, 0], bounds[axis_idx, 1]
                tol_low = tolerance + abs(low) * BOUNDARY_REL_TOL
                tol_high = tolerance + abs(high) * BOUNDARY_REL_TOL
                if abs(point[axis_idx] - low) <= tol_low:
                    return BoundaryFace(axis_idx, "min")
                if abs(point[axis_idx] - high) <= tol_high:
                    return BoundaryFace(axis_idx, "max")
            # Not on any axis-aligned face; fall through to SDF only if
            # bounds came from self (i.e., caller hasn't asserted bounds-only).
            if domain_bounds is not None:
                return None

        # Method 2: SDF domain (normal-based detection)
        if self.domain_sdf is not None:
            phi = self.domain_sdf(point)
            if abs(phi) > tolerance:
                return None

            normal = _compute_sdf_gradient(point, self.domain_sdf, epsilon=1e-5)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-12:
                return BoundaryFace(0, "min")  # Degenerate case fallback

            normal = normal / normal_norm
            dominant_axis = int(np.argmax(np.abs(normal)))
            side = "max" if normal[dominant_axis] > 0 else "min"
            return BoundaryFace(dominant_axis, side)

        if bounds is None:
            raise ValueError("Either domain_bounds or domain_sdf must be set")
        return None

    def outward_normal_for_face(
        self,
        face: BoundaryFace,
        dimension: int | None = None,
    ) -> np.ndarray:
        """Outward unit normal for an axis-aligned boundary face.

        Pure function of the face — no SDF gradient, no tolerance, no
        ambiguity. Use this when the caller has already classified the
        point to a face (e.g., via :meth:`identify_boundary_face`):
        avoids re-running classification and avoids the SDF-gradient path
        which mis-fires on Difference-style domains where ``domain_sdf``
        is the *obstacle* SDF rather than the outer box's.

        Args:
            face: BoundaryFace(axis, side) the point lies on.
            dimension: Optional dimension override. Defaults to
                ``self.dimension``.

        Returns:
            Unit outward normal vector of shape (dimension,). Outward
            means *away from the interior* — for an axis-aligned face,
            ``normal[axis] = -1`` if side is "min", ``+1`` if "max"; all
            other entries are zero.
        """
        d = dimension if dimension is not None else self._require_dimension("outward_normal_for_face")
        normal = np.zeros(d, dtype=float)
        normal[face.axis] = 1.0 if face.side == "max" else -1.0
        return normal

    def identify_boundary_id(self, point: np.ndarray, tolerance: float = BOUNDARY_TOL) -> str | None:
        """
        Identify which boundary a point lies on (legacy string interface).

        For rectangular domains, returns axis-aligned boundary IDs (e.g., "x_min", "y_max").
        Delegates to identify_boundary_face() and converts to string.

        Args:
            point: Spatial coordinates
            tolerance: Tolerance for boundary detection (default 1e-6, matched
                to identify_boundary_face).

        Returns:
            Boundary identifier string or None if not on boundary
        """
        face = self.identify_boundary_face(point, tolerance)
        if face is None:
            return None
        return face.to_string()

    def _normal_to_boundary_id(self, normal: np.ndarray) -> str:
        """Map outward normal vector to a boundary identifier string (legacy)."""
        dominant_axis = int(np.argmax(np.abs(normal)))
        side = "max" if normal[dominant_axis] > 0 else "min"
        return BoundaryFace(dominant_axis, side).to_string()

    def is_on_boundary(self, point: np.ndarray, tolerance: float = SDF_BOUNDARY_TOL) -> bool:
        """
        Check if a point is on the domain boundary.

        Args:
            point: Spatial coordinates
            tolerance: Tolerance for boundary detection

        Returns:
            True if point is on the boundary
        """
        dimension = self._require_dimension("is_on_boundary")
        point = np.asarray(point, dtype=float)

        # Rectangular domain: check if on any axis boundary
        if self.domain_bounds is not None:
            for axis_idx in range(dimension):
                if abs(point[axis_idx] - self.domain_bounds[axis_idx, 0]) < tolerance:
                    return True
                if abs(point[axis_idx] - self.domain_bounds[axis_idx, 1]) < tolerance:
                    return True
            return False

        # SDF domain: check if |phi| < tolerance
        if self.domain_sdf is not None:
            phi = self.domain_sdf(point)
            return abs(phi) < tolerance

        return False

    def get_outward_normal(self, point: np.ndarray, epsilon: float = 1e-5) -> np.ndarray | None:
        """
        Get the outward normal at a boundary point.

        Single classifier (Issue #1114): the exact axis-aligned face normal is preferred for
        points on an outer-box wall, and the SDF gradient is used only for genuinely curved
        boundaries (or pure-SDF domains with no box). The prior ordering checked ``domain_sdf``
        first, which mis-fired on Difference-style domains where ``domain_sdf`` is the *obstacle's*
        SDF: a point on the outer wall received the obstacle-pointing gradient (off by tens of
        degrees) instead of the wall normal. Routing the face case through
        :meth:`outward_normal_for_face` unifies the two normal sources.

        Args:
            point: Spatial coordinates on the boundary
            epsilon: Finite difference step for SDF gradient

        Returns:
            Unit outward normal vector, or None if not available
        """
        dimension = self._require_dimension("get_outward_normal")
        point = np.asarray(point, dtype=float)

        # Outer-box wall: exact face-derived normal (the canonical source). Checked first so an
        # obstacle SDF cannot override the wall normal on a Difference-style domain. Only an
        # axis-aligned bound match counts — NOT identify_boundary_face, whose SDF Method-2 also
        # classifies curved obstacle-surface points (those must keep the SDF gradient below).
        if self.domain_bounds is not None:
            face = self.identify_boundary_face(point)
            if face is not None:
                low, high = self.domain_bounds[face.axis]
                bound = high if face.side == "max" else low
                if abs(float(point[face.axis]) - bound) <= BOUNDARY_TOL + abs(bound) * BOUNDARY_REL_TOL:
                    return self.outward_normal_for_face(face, dimension=dimension)

        # Curved / SDF boundary (e.g. the obstacle surface, or a pure-SDF domain): use gradient.
        if self.domain_sdf is not None:
            normal = _compute_sdf_gradient(point, self.domain_sdf, epsilon=epsilon)
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-12:
                return normal / normal_norm
            return None

        return None

    # =========================================================================
    # Flux-Limited Absorption (for DIRICHLET exits)
    # =========================================================================

    def has_flux_limits(self) -> bool:
        """Check if any segment has flux capacity limits."""
        return any(seg.flux_capacity is not None for seg in self.segments)

    def get_flux_limits(self) -> dict[str, float]:
        """
        Get flux capacities for all segments that have limits.

        Returns:
            Dict mapping segment name to flux capacity (mass/time or particles/time).
            Only includes segments with explicit flux_capacity set.

        Example:
            >>> bc = mixed_bc(segments=[
            ...     BCSegment("exit_A", BCType.DIRICHLET, flux_capacity=0.1),
            ...     BCSegment("exit_B", BCType.DIRICHLET, flux_capacity=0.2),
            ...     BCSegment("walls", BCType.NO_FLUX),  # No flux limit
            ... ], dimension=2)
            >>> bc.get_flux_limits()
            {'exit_A': 0.1, 'exit_B': 0.2}
        """
        return {seg.name: seg.flux_capacity for seg in self.segments if seg.flux_capacity is not None}

    def get_flux_limit_for_segment(self, name: str) -> float | None:
        """Get flux capacity for a specific segment by name."""
        for seg in self.segments:
            if seg.name == name:
                return seg.flux_capacity
        return None

    def compute_particle_flux_limits(
        self,
        dt: float,
        n_particles: int,
        total_mass: float = 1.0,
    ) -> dict[str, int]:
        """
        Convert mass-based flux capacities to particle counts for a timestep.

        For particle methods, flux_capacity is in mass/time units.
        This converts to max particles per timestep.

        Args:
            dt: Timestep duration
            n_particles: Total number of particles in simulation
            total_mass: Total mass represented by particles (default 1.0)

        Returns:
            Dict mapping segment name to max particles absorbed per timestep.

        Example:
            >>> # flux_capacity=0.1 means 10% of total mass can exit per unit time
            >>> bc.compute_particle_flux_limits(dt=0.1, n_particles=1000, total_mass=1.0)
            {'exit_A': 10}  # 0.1 * 0.1 * 1000 = 10 particles
        """
        mass_per_particle = total_mass / n_particles
        limits = {}

        for seg in self.segments:
            if seg.flux_capacity is not None:
                # flux_capacity * dt = mass that can exit this timestep
                # Divide by mass_per_particle = max particles
                max_particles = int(seg.flux_capacity * dt / mass_per_particle)
                limits[seg.name] = max(1, max_particles)  # At least 1 if any capacity

        return limits

    def validate_values(self) -> None:
        """
        Validate that required values are provided for boundary condition segments.

        For uniform BCs, checks that the segment has appropriate values set.
        For mixed BCs, validates each segment individually.

        Raises:
            ValueError: If required values are missing for a BC type.

        """
        for segment in self.segments:
            bc_type = segment.bc_type

            if bc_type in (BCType.DIRICHLET, BCType.NEUMANN, BCType.NO_FLUX):
                # These types require a value (can be 0.0 which is valid)
                if segment.value is None:
                    raise ValueError(f"Segment '{segment.name}' with {bc_type.value} BC requires a value")

            elif bc_type == BCType.ROBIN:
                # Robin requires alpha, beta, and value
                if segment.alpha is None or segment.beta is None:
                    raise ValueError(f"Segment '{segment.name}' with Robin BC requires alpha and beta coefficients")
                if segment.value is None:
                    raise ValueError(f"Segment '{segment.name}' with Robin BC requires a value")

            # Periodic BC doesn't require values - it's handled by wrapping

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the mixed BC configuration.

        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []

        # Check that dimension is set for full validation
        if self.dimension is None:
            warnings.append("Dimension not set. Some validation checks skipped.")

        # Check that at least one domain specification exists
        if self.domain_bounds is None and self.domain_sdf is None:
            warnings.append("Neither domain_bounds nor domain_sdf is set")

        # Check segments have valid dimension (for rectangular domains) - only if dimension is set
        if self.dimension is not None:
            for segment in self.segments:
                if segment.region is not None:
                    max_axis = max(
                        (k if isinstance(k, int) else 0 for k in segment.region),
                        default=-1,
                    )
                    if max_axis >= self.dimension:
                        warnings.append(f"Segment '{segment.name}' region exceeds dimension {self.dimension}")

                # Check normal_direction dimension
                if segment.normal_direction is not None:
                    if len(segment.normal_direction) != self.dimension:
                        warnings.append(
                            f"Segment '{segment.name}' normal_direction has wrong dimension: "
                            f"expected {self.dimension}, got {len(segment.normal_direction)}"
                        )

        # Check for conflicting segments with same priority
        priority_groups = {}
        for segment in self.segments:
            if segment.priority not in priority_groups:
                priority_groups[segment.priority] = []
            priority_groups[segment.priority].append(segment)

        for priority, group in priority_groups.items():
            if len(group) > 1:
                warnings.append(f"Multiple segments with priority {priority}: {[s.name for s in group]}")

        # Check boundary coverage for rectangular domains - only if dimension is set
        # Warn if no segments cover certain boundaries (may indicate incomplete BC specification)
        if self.dimension is not None and self.domain_bounds is not None and not self.is_uniform:
            # Map axis index to standard names
            for axis_idx in range(self.dimension):
                # Check if min and max boundaries on this axis have at least one segment.
                # Uses BoundaryFace for dimension-agnostic matching (Issue #946).
                target_min = BoundaryFace(axis_idx, "min")
                target_max = BoundaryFace(axis_idx, "max")

                def _seg_covers_face(seg: BCSegment, target: BoundaryFace) -> bool:
                    """Check if a segment covers a specific boundary face."""
                    if seg.boundary is None or seg.boundary == "all":
                        return True
                    face = seg.face
                    return face is not None and face == target

                has_min_coverage = any(_seg_covers_face(seg, target_min) for seg in self.segments)
                has_max_coverage = any(_seg_covers_face(seg, target_max) for seg in self.segments)

                # If no explicit segment coverage, default BC will be used.
                # Issue #1100: default_bc may be None (unspecified) -> flag that
                # resolution will fail loud at those points rather than crashing here.
                face_label_min = target_min.to_string()
                face_label_max = target_max.to_string()
                default_desc = self.default_bc.value if self.default_bc is not None else "unspecified (will raise)"
                if not has_min_coverage:
                    warnings.append(
                        f"No explicit BC segment for {face_label_min} boundary. Default BC ({default_desc}) will be used."
                    )
                if not has_max_coverage:
                    warnings.append(
                        f"No explicit BC segment for {face_label_max} boundary. Default BC ({default_desc}) will be used."
                    )

        is_valid = len(warnings) == 0
        return is_valid, warnings

    def __str__(self) -> str:
        """String representation."""
        dim_str = f"{self.dimension}D" if self.dimension is not None else "unbound"
        # Structural, not the owner's answer: a description must not raise the owner's refusals.
        if self.is_uniform:
            seg = self.segments[0]
            return f"BoundaryConditions({dim_str}, {seg.bc_type.value}, value={seg.value})"

        # Mixed BC
        domain_type = "rectangular" if self.domain_bounds is not None else "SDF"
        lines = [f"BoundaryConditions({dim_str}, mixed, {domain_type}):"]
        for segment in self.segments:
            lines.append(f"  - {segment}")
        # Issue #1100: default_bc may be None (unspecified) -> guard .value access.
        default_desc = self.default_bc.value if self.default_bc is not None else "unspecified"
        lines.append(f"  - Default: {default_desc} = {self.default_value}")
        if self.corner_strategy != "priority":
            lines.append(f"  - Corner handling: {self.corner_strategy}")
        return "\n".join(lines)

    def declares_the_same_as(self, other: BoundaryConditions) -> bool:
        """Do two BCs declare one condition, whatever binding to a grid has added (#2530, row B9)?

        A problem's components BC and its geometry's are two declarations of one shared BC, and they agree when
        every declared field is equal as stored: the segments field by field, ``default_bc``, ``default_value``,
        ``domain_bounds``, ``domain_sdf``, ``corner_strategy`` and ``corner_mollification_radius``. ``dimension``
        and ``periodic_convention``, which binding fills in, agree when they are equal or one side leaves them
        unset. The grid reference is not compared, and identity is not the test: a grid stores a bound copy of the
        BC it is given.
        """
        if len(self.segments) != len(other.segments):
            return False
        segment_fields = [f.name for f in fields(BCSegment)]
        for mine, theirs in zip(self.segments, other.segments, strict=True):
            if not all(_equal_as_stored(getattr(mine, name), getattr(theirs, name)) for name in segment_fields):
                return False
        declared = ("default_bc", "default_value", "domain_bounds", "domain_sdf", "corner_strategy")
        if not all(_equal_as_stored(getattr(self, name), getattr(other, name)) for name in declared):
            return False
        if not _equal_as_stored(self.corner_mollification_radius, other.corner_mollification_radius):
            return False
        return all(
            getattr(self, name) is None
            or getattr(other, name) is None
            or _equal_as_stored(getattr(self, name), getattr(other, name))
            for name in ("dimension", "periodic_convention")
        )


# =============================================================================
# Factory Functions for Boundary Conditions
# =============================================================================


def uniform_bc(
    bc_type: str | BCType,
    value: float | Callable = 0.0,
    dimension: int | None = None,
    domain_bounds: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> BoundaryConditions:
    """
    Create uniform boundary conditions (same type on all boundaries).

    Args:
        bc_type: BC type ("periodic", "dirichlet", "neumann", "robin", "no_flux")
        value: BC value (constant, or a callable g(t, x) named by its parameters: (t, x), (x) or (t))
        dimension: Spatial dimension. If None, dimension will be inferred when
            BC is attached to a Geometry (lazy binding).
        domain_bounds: Optional domain bounds array (dimension, 2)
        alpha: Robin coefficient for u term (only for Robin BC)
        beta: Robin coefficient for du/dn term (only for Robin BC)

    Returns:
        BoundaryConditions with single uniform segment
    """
    if isinstance(bc_type, str):
        bc_type = BCType(bc_type.lower())

    segment = BCSegment(
        name="uniform",
        bc_type=bc_type,
        value=value,
        alpha=alpha,
        beta=beta,
        priority=0,
    )
    return BoundaryConditions(
        dimension=dimension,
        segments=[segment],
        domain_bounds=domain_bounds,
        default_bc=bc_type,
        default_value=value if not callable(value) else 0.0,
    )


def periodic_bc(
    dimension: int | None = None,
    domain_bounds: np.ndarray | None = None,
    convention: PeriodicGridConvention | None = None,
) -> BoundaryConditions:
    """
    Create periodic boundary conditions.

    Args:
        dimension: Spatial dimension. If None, dimension will be inferred when
            BC is attached to a Geometry (lazy binding).
        domain_bounds: Optional domain bounds
        convention: Where the last node sits (Issue #1822). ``None`` leaves it unstated, which
            wraps as this package always has -- ``np.linspace(lo, hi, N, endpoint=False)``, all N
            nodes distinct. Attaching this BC to a ``TensorProductGrid`` binds the convention that
            grid measures from its own coordinates (``ENDPOINT_INCLUSIVE``: ``x[0]`` and ``x[-1]``
            are one physical point), so solver paths need not state it. State it here for a grid
            the BC never meets -- the operator layer holds no grid object. The two differ by
            exactly one node, and a wrap under the wrong one returns a finite, plausible field
            with every stencil shifted a cell; a stated convention that contradicts the grid it is
            attached to is refused rather than silently overridden.

    Returns:
        Uniform periodic BC
    """
    bc = uniform_bc(BCType.PERIODIC, value=0.0, dimension=dimension, domain_bounds=domain_bounds)
    bc.periodic_convention = convention
    return bc


def periodic_faces(bc: Any, dimension: int, geometry: Any = None) -> list[bool]:
    """Whether each face of ``bc`` is PERIODIC, axis by axis and min before max. Issue #2495.

    Read through the face owner, :meth:`BoundaryConditions.segment_for_face`, asked by `BoundaryFace` and with
    periodicity as the equivalence: a segment covering part of a face is accepted when it agrees with the rest
    of the face on that, and refused otherwise (#2512 row B1). ``geometry`` is the grid, on which the owner
    reads a ``region``, ``sdf_region`` or ``region_name`` placement. The one face loop both
    :func:`periodic_on_every_face` and FP-FDM's refusal read.
    """

    def same_periodicity(first: BCSegment, second: BCSegment) -> bool:
        return (first.bc_type is BCType.PERIODIC) == (second.bc_type is BCType.PERIODIC)

    return [
        bc.segment_for_face(BoundaryFace(axis, side), geometry, equivalent=same_periodicity).bc_type is BCType.PERIODIC
        for axis in range(dimension)
        for side in ("min", "max")
    ]


def periodic_on_every_face(bc: Any, dimension: int | None = None, geometry: Any = None) -> bool:
    """Whether every face of ``bc`` is periodic, however the periodicity is spelt. Issue #2495.

    A uniform periodic BC, one PERIODIC segment per face, and an empty segment list over a periodic
    ``default_bc`` all say the same thing. The test this replaced, ``bc.is_uniform and bc.type ==
    "periodic"``, accepted only the first; FP-FDM then solved the other two as no-flux, bit for bit,
    with no error raised.

    The faces are read through :func:`periodic_faces`, that is through the face owner, on ``geometry`` when
    given (#2512 row B1). ``dimension`` is the domain's, for a BC not yet bound to one; without either, there
    are no faces to read and the answer is ``False``.
    """
    from .bc_utils import declares_periodic

    if bc.is_uniform:
        return bc.type == "periodic"
    dimension = dimension or getattr(bc, "dimension", None)
    if not declares_periodic(bc) or not dimension:
        return False
    return all(periodic_faces(bc, dimension, geometry))


def periodic_axis_span(bc: Any, n: int, dimension: int | None = None, geometry: Any = None) -> int | None:
    """How many DISTINCT cells an axis of ``n`` nodes has, or ``None`` if it does not wrap.

    Issue #1822. A wrap needs two facts, and reading only the first is what every periodic
    stencil in this package got wrong: ``bc`` says the axis wraps, and ``bc.periodic_convention``
    says whether the last node is a *new* cell or the first one under another index. Under
    ``ENDPOINT_INCLUSIVE`` -- what ``TensorProductGrid`` builds -- it is the latter, so ``i-1``
    from node 0 is node ``n-2`` and the modular arithmetic must divide by ``n-1``.

    Dividing by ``n`` there solves the problem on a torus one cell too long. Measured on the
    FP-FDM family at ``Nx=21``: 8.7e-02 of relative error against the analytic heat kernel, where
    the same scheme on the equivalent ``n-1``-cell torus gives 9.3e-03. Nothing raises either way.

    Returns ``None`` -- not ``n`` -- for a non-periodic axis, so a caller cannot use the number
    without having decided what to do about the wrap.
    """
    if not isinstance(bc, BoundaryConditions):
        return None  # not a BoundaryConditions: FP-FDM's dispatch refuses it (#1559)
    if not periodic_on_every_face(bc, dimension, geometry):
        return None
    return n - repeated_endpoint_count(getattr(bc, "periodic_convention", None))


def repeated_endpoint_mirror(
    bc: Any, multi_idx: tuple[int, ...], shape: tuple[int, ...], geometry: Any = None
) -> tuple[int, ...] | None:
    """The node ``multi_idx`` duplicates under an inclusive wrap, or ``None`` if it duplicates none.

    Issue #1822. On an inclusive periodic axis the last node is not an unknown: it is node 0 under
    another index. An assembly that gives it a stencil row of its own is solving for one cell too
    many, and the duplicated pair then drifts apart because their neighbourhoods differ.
    """
    mirrored = list(multi_idx)
    duplicated = False
    for d in range(len(shape)):
        span = periodic_axis_span(bc, shape[d], len(shape), geometry)
        if span is not None and span < shape[d] and multi_idx[d] >= span:
            mirrored[d] = multi_idx[d] - span
            duplicated = True
    return tuple(mirrored) if duplicated else None


def dirichlet_bc(
    value: float | Callable = 0.0,
    dimension: int | None = None,
    domain_bounds: np.ndarray | None = None,
) -> BoundaryConditions:
    """
    Create Dirichlet boundary conditions (u = value at boundary).

    Args:
        value: Boundary value (constant, or a callable g(t, x) named by its parameters: (t, x), (x) or (t))
        dimension: Spatial dimension. If None, dimension will be inferred when
            BC is attached to a Geometry (lazy binding).
        domain_bounds: Optional domain bounds

    Returns:
        Uniform Dirichlet BC
    """
    return uniform_bc(BCType.DIRICHLET, value=value, dimension=dimension, domain_bounds=domain_bounds)


def neumann_bc(
    value: float | Callable = 0.0,
    dimension: int | None = None,
    domain_bounds: np.ndarray | None = None,
) -> BoundaryConditions:
    """
    Create Neumann boundary conditions (du/dn = value at boundary).

    Two readings, by who reads the BC (#2512, row B3):

    - As the problem's shared BC, ``value`` is the HJB's: du/dn = g, the boundary cost per unit of
      boundary local time. With g = 0 the FP reads the face as zero total flux, J.n = 0 (ruled
      2026-10-08). A nonzero g is refused at the FP (user ruling 2026-10-09): it says nothing about the
      agents' mass at the wall, so keep ``neumann_bc(value=g)`` on the problem for the HJB and give the
      FP solver ``boundary_conditions=no_flux_bc(...)``.
    - Handed to an FP solver explicitly it would mean dm/dn = g, which no FP solver implements, so the
      solver refuses it. Pass ``no_flux_bc()`` for a reflecting FP wall.

    Args:
        value: Normal derivative value (constant, or a callable g(t, x) named by its parameters: (t, x), (x) or (t))
        dimension: Spatial dimension. If None, dimension will be inferred when
            BC is attached to a Geometry (lazy binding).
        domain_bounds: Optional domain bounds

    Returns:
        Uniform Neumann BC
    """
    return uniform_bc(BCType.NEUMANN, value=value, dimension=dimension, domain_bounds=domain_bounds)


def no_flux_bc(dimension: int | None = None, domain_bounds: np.ndarray | None = None) -> BoundaryConditions:
    """
    Create no-flux boundary conditions.

    The condition differs by equation, and both are what "no flux" should mean on that side:

    - **FP side**: zero total flux, ``J.n = 0`` with ``J = v*m - D*grad(m)``. This is the
      mass-conserving wall, and it is what the divergence-form discretisation enforces. With
      drift at the boundary it implies ``D dm/dn = (v.n) m``, so the normal derivative of the
      density is generally **not** zero.
    - **HJB side**: zero normal derivative, ``du/dn = 0`` -- the adjoint of a reflecting wall.

    Common for Fokker-Planck equations, where it is the correct choice for conserving mass.

    .. note::
       ``NoFluxCalculator`` is a deprecated alias for ``ZeroGradientCalculator`` (``du/dn = 0``)
       and is **not** the flux condition described here; use ``ZeroFluxCalculator`` if selecting a
       calculator directly.

    Args:
        dimension: Spatial dimension. If None, dimension will be inferred when
            BC is attached to a Geometry (lazy binding).
        domain_bounds: Optional domain bounds

    Returns:
        Uniform no-flux BC
    """
    return uniform_bc(BCType.NO_FLUX, value=0.0, dimension=dimension, domain_bounds=domain_bounds)


def robin_bc(
    value: float | Callable = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    dimension: int | None = None,
    domain_bounds: np.ndarray | None = None,
) -> BoundaryConditions:
    """
    Create Robin boundary conditions (alpha*u + beta*du/dn = value).

    That is the HJB's reading. Handed to an FP solver as its own BC, the coefficients mean a condition on
    the total flux, ``J.n = (D/beta)*(alpha*m - value)``; on a BC the two equations share, a ROBIN is the
    HJB's and is refused at the FP (#2512, user ruling 2026-10-10; ``docs/user/CONVENTIONS.md``).

    **You probably do not want this for a reflecting FP wall.** ``J.n = 0`` is Robin in ``m``, but
    the conservative schemes already impose it structurally -- by zeroing the total face flux,
    with no BC-type branch naming it -- and ``FPParticleSolver`` gets the same wall from Skorokhod
    reflection. Adding a Robin segment on top of such a wall destroys it rather than restating it.

    Which solvers read the coefficients (#1975):

    - ``FPFEMSolver`` (from its own ``boundary_conditions`` only) / ``HJBFEMSolver`` -- weak form,
      coefficients read:
      ``A_robin = D*(alpha/beta)*int_dOmega phi_i phi_j``, load ``D*(1/beta)*int_dOmega g phi_i``.
      Constant ``g`` only; ``beta == 0`` fails loud; a provider-valued ``alpha`` raises a bare
      ``TypeError`` from ``float()``.
    - ``HJBGFDMSolver`` -- the ``Robin(0, 1)`` case only, i.e. ``n . grad u = g``.
    - **Every grid FP solver refuses ROBIN at construction**, uniform and mixed alike: a shared one in
      ``fp_view_of_shared_bc`` (#2512), its own in ``_validate_bc_support`` (#1456, raising from
      ``BaseMFGSolver``). The refusal is load-bearing:
      the FDM boundary handlers are not passed ``boundary_conditions``, so they read none of
      ``alpha``/``beta``/``value``. Below the gate -- calling ``solve_timestep_full_nd`` directly,
      or mutating ``solver.boundary_conditions`` after construction (#2475) -- a ROBIN segment is
      byte-identical to no-flux, and a provider-valued coefficient is accepted silently (#1979).

    The perturbation a Robin segment adds to an already-reflecting wall:

        J.n = D*(alpha/beta)*m          i.e.   D d_n m = (v_n - D*alpha/beta) * m

    ``A_robin``'s boundary column sums are exactly ``D*alpha/beta``: measured 0.1440 / 0.4000 /
    1.6000 / 3.2000 at beta=1 for D = 0.045 / 0.125 / 0.5 / 1.0, and 0.0360 at
    (D, alpha, beta) = (0.045, 3.2, 4) where ``D*alpha`` would be 0.1440. **It scales with ``D``
    and inversely with ``beta``, never with ``v_n``** -- so the invariant is the ratio, and any
    test written on ``alpha`` alone passes a whole equivalence class of identical walls.
    Measured: ``(-2*v_n, 2*D)`` is bit-identical to ``(-v_n, D)`` (``max|m - m_ref| = 0``), while
    ``(-v_n, 2*D)`` differs by ``6.2e+46``.

    **The trap.** The reflecting condition's own coefficients are
    ``(alpha, beta) = (D_pH(x, grad u).n, D)``, so ``alpha = -v_n``, NOT ``+v_n``, wherever the FP
    transport velocity is ``v = -D_pH``. That also follows from the library's own
    ``J = v*m - D grad m`` without mentioning ``D_pH``, which is the sense-free route to it. Those are the
    coefficients of the reflecting wall read as a condition on ``m``. An FP solver reads a ROBIN as a
    condition on the flux, ``J.n = (D/beta)(alpha*m - g)`` (#2512, user ruling 2026-10-10), so there the
    reflecting wall is ``ROBIN(0, beta, 0)``, which is ``NO_FLUX``.

    ``D*alpha/beta = -v_n`` is by the law above the row that DOUBLES -- an influx. So encoding the
    reflecting condition as a Robin segment on a conservative assembly is unbounded, not merely
    leaky. (Magnitudes are not quotable: over mesh 50-400, T 0.05-0.4, Nt 50-800 and IC width
    0.05-0.3 the growth spans well over ninety orders of magnitude and flips sign at coarse
    ``Nt``, where the positivity clip dominates -- ``weak_form_fp_solver.py:225-230`` reports its
    cumulative injection at solve end.)

    Measured wall, ``sigma=0.3``, ``v_n=+3.2``, ``D=0.045`` (so ``v_n/D = 71.111``), ``FPFDMSolver``
    / ``divergence_upwind`` on [0,1], Gaussian initial density of width 0.1 at 0.5, T=0.2, Nt=200,
    ``drift_field=+3.2``. Columns 3 and 4 are ``alpha*m + beta*d_n m`` normalised by ``|alpha|*m``,
    i.e. column 2 rearranged rather than a second measurement:

        Nx      d_n m / m     (+v_n, D)     (-v_n, D)
        161        56.41        1.7933       -0.2067
        321        63.27        1.8898       -0.1102
        641        67.10        1.9437       -0.0563
       1281        69.09        1.9716       -0.0284

    Column 2 converges to 71.111 at first order; ``(-v_n, D)`` goes to zero and ``(+v_n, D)`` to 2.

    The ``v = -D_pH`` antecedent now holds unconditionally: the library is minimisation-only
    (#2373). ``HamiltonianBase.optimal_control`` returns ``-dH_dp``, and ``SeparableHamiltonian``
    overrides it to delegate to ``control_cost.optimal_control(p)``, since in the separable case
    the control enters only through the control cost. Before #2373 each route carried its own
    ``sense`` field and they could be set to disagree with nothing checking; removing the concept
    removed that gap rather than guarding it. Paths that form the drift themselves are still gated
    to a QUADRATIC control cost by ``assert_quadratic_drift`` -- the remaining refusal is
    about the wrong FORM (a regularised cost), not a wrong direction.

    For contrast, the ``gradient_*`` family, removed in #2007, imposed ``d_n m = 0`` by hard-coding the
    mirrored ghost ``m_{N+1} = m_{N-1}`` and was non-conservative by design (#1075). How much it
    leaked was a property of the configuration, not of the family -- at sigma=0.3, 81 points on
    [0,1], dt=1e-3, Gaussian initial density of width s0 at 0.5, potential channel:

        scheme               s0     T=0.20    T=0.30    T=0.50
        gradient_centered    0.1     -78.1%    -99.0%    -100.0%
        gradient_upwind      0.1     -75.5%    -97.9%    -100.0%
        gradient_centered    0.3     -42.0%    -57.6%     -60.0%

    **Reach for ``robin_bc`` when you want a wall that is not the reflecting one:
    ``D*alpha/beta != -v_n``, or an inhomogeneous ``g``.** See #1975.

    Args:
        value: RHS value g in alpha*u + beta*du/dn = g
        alpha: Coefficient of u
        beta: Coefficient of du/dn
        dimension: Spatial dimension. If None, dimension will be inferred when
            BC is attached to a Geometry (lazy binding).
        domain_bounds: Optional domain bounds

    Returns:
        Uniform Robin BC
    """
    return uniform_bc(
        BCType.ROBIN,
        value=value,
        dimension=dimension,
        domain_bounds=domain_bounds,
        alpha=alpha,
        beta=beta,
    )


@deprecated(
    since="v0.18.0",
    replacement="Use BoundaryConditions(segments=[...]) directly",
    reason="Factory is redundant - direct construction is clearer",
)
def mixed_bc(
    segments: list[BCSegment],
    dimension: int | None = None,
    domain_bounds: np.ndarray | None = None,
    domain_sdf: Callable[[np.ndarray], float] | None = None,
    default_bc: BCType = BCType.NEUMANN,
    default_value: float = 0.0,
    corner_strategy: Literal["priority", "average", "mollify"] = "priority",
) -> BoundaryConditions:
    """
    DEPRECATED: Use BoundaryConditions(segments=[...]) directly.

    Migration:
        # Old
        bc = mixed_bc([seg1, seg2], dimension=2, domain_bounds=bounds)

        # New (preferred)
        bc = BoundaryConditions(segments=[seg1, seg2], dimension=2, domain_bounds=bounds)
    """
    return BoundaryConditions(
        dimension=dimension,
        segments=segments,
        domain_bounds=domain_bounds,
        domain_sdf=domain_sdf,
        default_bc=default_bc,
        default_value=default_value,
        corner_strategy=corner_strategy,
    )


def _region_faces(mask: np.ndarray) -> tuple[set[BoundaryFace], set[BoundaryFace]]:
    """The faces a grid-shaped region mask covers, and the faces it covers only part of. #2472

    A face counts as covered when the region holds all of its interior: the face without the
    points it shares with other faces, so a region on ``x_max`` does not claim ``y_min`` through
    their common corner. On an axis with two points a face has no interior, since every point of it
    is shared, and then it counts as covered only if the region holds the whole face.
    """
    covered: set[BoundaryFace] = set()
    partial: set[BoundaryFace] = set()
    for axis in range(mask.ndim):
        for side, index in (("min", 0), ("max", -1)):
            face_mask = np.take(mask, index, axis=axis)
            interior = face_mask[tuple(slice(1, -1) for _ in range(face_mask.ndim))]
            if interior.size == 0:
                if face_mask.all():
                    covered.add(BoundaryFace(axis, side))
            elif interior.all():
                covered.add(BoundaryFace(axis, side))
            elif interior.any():
                partial.add(BoundaryFace(axis, side))
    return covered, partial


def _is_global(segment: BCSegment) -> bool:
    """A GLOBAL segment ("the rest"): no placement field, or ``boundary="all"`` and nothing else placing it."""
    return (
        segment.boundary in (None, "all")
        and segment.region is None
        and segment.region_name is None
        and segment.sdf_region is None
        and segment.normal_direction is None
    )


def _segments_governing_nothing(
    order: list[BCSegment],
    governs: set[int],
    touched: dict[int, list[tuple[str, BCSegment]]],
    partial_ahead: set[int],
    winners: list[BCSegment],
    same: Callable[[BCSegment, BCSegment], bool],
) -> tuple[list[BCSegment], list[tuple[BCSegment, list[tuple[str, BCSegment]]]]]:
    """The segments that govern no location and are not effect-neutral, for the shadow refusal (#2512 row B1).

    ``touched`` gives, for each segment that governs nowhere, where it is outranked and by what. One that is outranked
    somewhere is dropped there unless ``same`` as each segment that outranks it; one that reaches nowhere (and covers
    no part of a face ahead of its winner, ``partial_ahead``) is dropped unless ``same`` as every winner. Returns the
    second kind, then the first with its outranking segments. Read by the box faces (`_check_shadowed`) and by a mesh's
    facets (`BoundaryConditions.segments_for_facets`).
    """
    nowhere, dead = [], []
    for seg in order:
        if id(seg) in governs:
            continue
        hits = touched[id(seg)]
        if not hits and id(seg) not in partial_ahead:
            if not all(same(seg, winner) for winner in winners):
                nowhere.append(seg)
            continue
        if all(same(seg, winner) for _, winner in hits):
            continue  # effect-neutral wherever it is shadowed, or a partial cover ahead of a winner
        dead.append((seg, hits))
    return nowhere, dead


def _equal_as_stored(a: Any, b: Any) -> bool:
    """Equal as stored: a callable or an object equals only itself, an array equals an array of the same shape and
    entries. The owner's comparison of declared values, for `_same_condition` and
    `BoundaryConditions.declares_the_same_as`."""
    if a is b:
        return True
    try:
        return bool(np.all(np.asarray(a) == np.asarray(b))) and np.shape(a) == np.shape(b)
    except (TypeError, ValueError):
        return False


def _same_condition(first: BCSegment, second: BCSegment) -> bool:
    """Do two segments impose the same condition: type, value, flux capacity and (for ROBIN) coefficients, by
    equality? The owner's one sameness rule: the shadow rule, the corner check and `uniform_segment` read it.

    Compared as stored: a callable or an array value equals only itself, so two segments that would
    give equal numbers through different objects are not taken to be the same condition. ``flux_capacity`` is
    part of the condition: two exits that differ only in capacity let out different mass.
    """
    if first.bc_type != second.bc_type or not _equal_as_stored(first.value, second.value):
        return False
    if not _equal_as_stored(first.flux_capacity, second.flux_capacity):
        return False
    if first.bc_type != BCType.ROBIN:
        return True
    from .types import segment_robin_coefficients  # the one reader of a Robin segment's coefficients (#2512 B2)

    where = "BoundaryConditions.segment_for_face (are two segments the same condition?)"
    return all(
        _equal_as_stored(a, b)
        for a, b in zip(
            segment_robin_coefficients(first, where=where, as_stored=True),
            segment_robin_coefficients(second, where=where, as_stored=True),
            strict=True,
        )
    )


def _region_axis(segment: BCSegment, key: str | int, dimension: int) -> int:
    """The axis index a ``region`` key names: an int, or one of x, y, z, w (as ``matches_point`` reads it)."""
    names = {"x": 0, "y": 1, "z": 2, "w": 3}
    axis = key if isinstance(key, int) else names.get(key)
    if axis is None or not 0 <= axis < dimension:
        raise ValueError(f"BC segment {segment.name!r}: region axis {key!r} names no axis of a {dimension}-D domain.")
    return axis


def _group_nodes(
    corners: dict[tuple[int, ...], tuple[BoundaryFace, Any]],
) -> list[tuple[BoundaryFace, Any, tuple[np.ndarray, ...]]]:
    """`BoundaryConditions.corner_nodes`' answer grouped by face and segment, each group with an index for fancy
    indexing, so a reader writes each group once."""
    groups: dict[tuple[BoundaryFace, int], tuple[BoundaryFace, Any, list[tuple[int, ...]]]] = {}
    for node, (face, segment) in corners.items():
        groups.setdefault((face, id(segment)), (face, segment, []))[2].append(node)
    return [(face, segment, tuple(np.array(nodes, dtype=np.intp).T)) for face, segment, nodes in groups.values()]


def _multi_face_nodes(shape: tuple[int, ...]):
    """Every node of a ``shape`` grid that lies on two or more faces, with those faces: the corners, and in 3-D the
    edges. Each node once, from the set of axes it sits at an end of."""
    ndim = len(shape)
    for count in range(2, ndim + 1):
        for axes in itertools.combinations(range(ndim), count):
            for sides in itertools.product(("min", "max"), repeat=count):
                fixed = dict(zip(axes, sides, strict=True))
                ranges = [
                    [0 if fixed[d] == "min" else shape[d] - 1] if d in fixed else range(1, shape[d] - 1)
                    for d in range(ndim)
                ]
                faces = tuple(BoundaryFace(d, fixed[d]) for d in axes)
                for node in itertools.product(*ranges):
                    yield tuple(int(i) for i in node), faces


def _placement_mask_on_grid(segment: BCSegment, geometry: Any) -> np.ndarray | None:
    """Which grid nodes a ``region``/``sdf_region`` placement holds, or None if ``geometry`` is no grid."""
    from mfgarchon.geometry.base import CartesianGrid  # local: geometry.base imports this package

    if not isinstance(geometry, CartesianGrid):
        return None
    grid: Any = geometry
    shape = tuple(grid.get_grid_shape())
    # The layout `_region_mask_on_grid` reads a marked region's mask in: the spatial grid, reshaped.
    points = np.asarray(grid.get_spatial_grid(), dtype=float).reshape(-1, len(shape))
    inside = np.ones(points.shape[0], dtype=bool)
    if segment.region is not None:
        for key, (low, high) in segment.region.items():
            axis = _region_axis(segment, key, len(shape))
            inside &= (points[:, axis] >= low - SDF_BOUNDARY_TOL) & (points[:, axis] <= high + SDF_BOUNDARY_TOL)
    if segment.sdf_region is not None:
        # One value per point, as `BCSegment.matches_point` reads it: a one-element array counts as its value.
        values = [np.asarray(segment.sdf_region(x), dtype=float) for x in points]
        if any(value.size != 1 for value in values):
            shapes = sorted({value.shape for value in values if value.size != 1})
            raise ValueError(
                f"BC segment {segment.name!r}: sdf_region returned arrays of shape {shapes} for single points; it must "
                "return one signed distance per point (#2512 row B1)."
            )
        inside &= np.array([value.item() <= SDF_BOUNDARY_TOL for value in values], dtype=bool)
    return inside.reshape(shape)


def _region_mask_on_grid(geometry: Any, region_name: str) -> np.ndarray | None:
    """The region's mask on the grid's index layout, or None if the geometry has no faces to read."""
    from mfgarchon.geometry.base import CartesianGrid  # local: geometry.base imports this package

    if not isinstance(geometry, CartesianGrid):
        return None
    marked: Any = geometry  # a CartesianGrid that also marks regions (SupportsRegionMarking)
    return np.asarray(marked.get_region_mask(region_name)).astype(bool).reshape(tuple(marked.get_grid_shape()))


def faces_covered_by_region(geometry: Any, region_name: str) -> frozenset[BoundaryFace]:
    """The boundary faces a marked region governs on a path that imposes one condition per face. #2472

    See :func:`_region_faces` for what "covered" means. A region covering part of a face's interior
    is refused: one condition per face cannot represent it, and either reading would silently drop
    the other. Whether a strip covers a face therefore depends on the resolution: a strip may hold
    a coarse face's whole interior and only part of a finer one's.

    Raises:
        ValueError: if the geometry is not a structured grid, if the region covers part of a face's
            interior, or if it covers no whole face.
    """
    mask = _region_mask_on_grid(geometry, region_name)
    if mask is None:
        raise ValueError(
            f"{type(geometry).__name__} has no grid faces, so marked region {region_name!r} cannot be "
            "resolved to a face (Issue #2472)."
        )
    covered, partial = _region_faces(mask)
    if partial:
        face = min(partial, key=lambda f: (f.axis, f.side))
        raise ValueError(
            f"Marked region {region_name!r} covers part of the {face.to_string()} face's interior. A "
            "face-level boundary condition imposes one condition per face and cannot represent a "
            "sub-face region. Mark the region on whole faces (mark_region(name, boundary=...)) or use a "
            "solver that resolves boundary conditions per point (Issue #2472)."
        )
    if not covered:
        raise ValueError(
            f"Marked region {region_name!r} covers no whole boundary face, so it cannot carry a boundary "
            "condition on a face-level path (Issue #2472)."
        )
    return frozenset(covered)


def _marks_region(geometry: Any, name: str) -> bool:
    """Whether ``geometry`` marks a region called ``name``. Then the region's mask places a ``region_name`` segment,
    whatever the name spells -- "x_max" included -- and a face label decides nothing (#2472, #2717)."""
    return isinstance(geometry, SupportsRegionMarking) and name in geometry.get_region_names()


def region_name_governs_face(segment: BCSegment, face: BoundaryFace, geometry: Any = None) -> bool:
    """Does a ``region_name`` segment govern ``face``? The one resolver for the face-level readers. #2472

    The geometry's region mask decides when the geometry is a structured grid that defines the region
    (:func:`faces_covered_by_region`). Without it, a region name that is itself a face label
    ("x_max", "left") names that face. Anything else is refused. Before, one reader let such a
    segment cover every face and another skipped it, so they gave different answers on the same
    object, and neither answer was the region. :func:`mixed_bc_from_regions` resolves whole-face
    regions to named faces when it builds the BC, so this is reached by a region it kept (one covering
    part of a face, or no whole face) or by a ``region_name`` segment built some other way.

    Raises:
        ValueError: if the region cannot be resolved to faces.
    """
    name = segment.region_name
    if _marks_region(geometry, name):
        return face in faces_covered_by_region(geometry, name)
    label = parse_boundary_face(name)
    if label is not None:
        return label == face
    raise ValueError(_region_name_refusal(segment, geometry, f"governs {face.to_string()}"))


def region_name_holds_point(
    segment: BCSegment,
    point: np.ndarray,
    boundary_id: str | None,
    domain_bounds: np.ndarray | None,
    tolerance: float,
    geometry: Any = None,
) -> bool:
    """Does a ``region_name`` segment hold the boundary point ``point``? The point-level reading of
    :func:`region_name_governs_face`, in its order (#2490 B1e-1):

    - a grid that marks the region decides by its mask, at the grid node nearest the point (the first of two
      equidistant ones in the grid's order). A particle past a wall reaches it at its nearest point of the BC's box,
      where `BoundaryConditions.segment_for_point` reads every point it reads on a face (#2780); a point outside
      ``domain_bounds`` read on no face is refused, since which face it crossed is unknown. A point
      outside the marking grid's own box, where the BC's box reaches past the grid's, or of another dimension than the
      grid, is refused, not read as outside the region; so is a point on a grid whose nodes do not strictly ascend on
      an axis, where the face rule reads the axis's first node as its min face whatever its coordinate (#2490);
    - otherwise a region name that is a face label ("x_max", "left") holds the points on that face: the face the
      point is read on (``boundary_id``), or, with none, the face of ``domain_bounds`` it lies on within ``tolerance``;
    - otherwise refused, as the face rule refuses.

    Before, the point rule required a geometry outright, so a label-named segment that governs its face was refused
    at a point on it; and a point it could not map was read as outside the region, by two ``except`` clauses, with no
    word.

    Raises:
        ValueError: where the region cannot be read at the point.
    """
    name = segment.region_name
    point = np.asarray(point, dtype=float)
    if _marks_region(geometry, name):
        if boundary_id is None and domain_bounds is not None and point.size == len(domain_bounds):
            box = np.asarray(domain_bounds, dtype=float)
            if np.any(point < box[:, 0] - tolerance) or np.any(point > box[:, 1] + tolerance):
                raise ValueError(
                    f"BC segment {segment.name!r} is restricted to the marked region {name!r}, and the point "
                    f"{point.tolist()} lies outside the box {box.tolist()} it is read in, on no face, so which face it "
                    "crossed is unknown (#2780). It is not read as outside the region: read it on the face it crossed, "
                    "segment_for_point(point, face), which reads it at its nearest point of the box."
                )
        node = _region_node(segment, geometry, point, tolerance)
        return bool(np.asarray(geometry.get_region_mask(name)).ravel()[node])
    label = parse_boundary_face(name)
    if label is None:
        raise ValueError(_region_name_refusal(segment, geometry, f"holds the boundary point {point.tolist()}"))
    if boundary_id is not None:
        on = parse_boundary_face(boundary_id)
        if on is not None:
            return on == label
    if domain_bounds is None:
        raise ValueError(
            f"BC segment {segment.name!r} is restricted to the face {name!r}, and the boundary point "
            f"{point.tolist()} is read with no face and no domain bounds, so which face it lies on is unknown "
            "(#2490). Read it with the face it lies on, or give the BC its domain_bounds."
        )
    side = np.asarray(domain_bounds, dtype=float)[label.axis, 1 if label.side == "max" else 0]
    return bool(abs(point[label.axis] - side) <= tolerance)


def _region_node(segment: BCSegment, geometry: Any, point: np.ndarray, tolerance: float) -> int:
    """The flat index, in the grid's C order, of the node of ``geometry`` nearest ``point``, for a region mask: refused
    where the geometry has no node coordinates, the point has another number of coordinates than the grid has axes, an
    axis's nodes do not strictly ascend, or the point lies outside the grid's own box by more than ``tolerance``."""
    region = f"BC segment {segment.name!r} is restricted to the marked region {segment.region_name!r}"
    coordinates = getattr(geometry, "coordinates", None)
    if coordinates is None:
        raise ValueError(
            f"{region}, and {type(geometry).__name__} gives no node coordinates to read its mask at the point "
            f"{point.tolist()} (#2490). Read the region on a TensorProductGrid, or place the segment by "
            "sdf_region=... or boundary=..."
        )
    if len(coordinates) != point.size:
        raise ValueError(
            f"{region}, and the point {point.tolist()} has {point.size} coordinates where the "
            f"{type(geometry).__name__} that marks it has {len(coordinates)} axes, so its mask cannot be read there "
            "(#2490). It is not read as outside the region: read the point in the grid's dimension."
        )
    box = geometry.get_bounds()
    lo, hi = box if box is not None else ([np.min(c) for c in coordinates], [np.max(c) for c in coordinates])
    index = []
    for axis, nodes in enumerate(coordinates):
        nodes = np.asarray(nodes, dtype=float)
        if not np.all(np.diff(nodes) > 0):
            mirrored = (
                "The face rule reads an axis's first node as its min face whatever its coordinate, and reading "
                "the point by coordinate would put the region on the other face. "
                if nodes[-1] < nodes[0]
                else ""
            )
            raise ValueError(
                f"{region}, and the nodes of the {type(geometry).__name__} that marks it do not strictly ascend on axis "
                f"{axis}, so its mask is not read at the point {point.tolist()} (#2490). {mirrored}Give the grid "
                "strictly ascending coordinates, with no repeated node."
            )
        if not (lo[axis] - tolerance <= point[axis] <= hi[axis] + tolerance):
            raise ValueError(
                f"{region}, and the point {point.tolist()} lies outside the grid that marks it, whose box on axis "
                f"{axis} is [{lo[axis]}, {hi[axis]}], so the region cannot be read there (#2490). It is not read as "
                "outside it: read the BC on the grid that marks the region, with domain_bounds equal to that grid's box."
            )
        index.append(int(np.argmin(np.abs(nodes - point[axis]))))
    return int(np.ravel_multi_index(tuple(index), tuple(len(nodes) for nodes in coordinates)))


def _region_name_refusal(segment: BCSegment, geometry: Any, what: str) -> str:
    """Why a ``region_name`` segment cannot be read: no grid the owner reads marks its region, and its name is no face
    label. The one message of the face rule and the point rule."""
    name = segment.region_name
    head = f"BC segment {segment.name!r} is restricted to the marked region {name!r}"
    get_region_names = getattr(geometry, "get_region_names", None)
    if callable(get_region_names) and name in get_region_names():
        return (
            f"{head}, and {type(geometry).__name__} marks a region of that name, but the boundary owner reads a "
            f"region by its mask on a grid's nodes, and {type(geometry).__name__} is no such grid (it does not "
            f"implement SupportsRegionMarking), so it cannot tell whether the segment {what} (#2490). Place the "
            "segment by sdf_region=..., a signed distance the owner reads at any point, or by boundary=..."
        )
    return (
        f"{head}, and which locations a region covers is known only to the geometry that marked it. "
        + (
            "This reader has no geometry"
            if geometry is None
            else f"{type(geometry).__name__} defines no region of that name"
        )
        + f", so it cannot tell whether the segment {what}. Build the BC with mixed_bc_from_regions(geometry, "
        "...), which resolves a whole-face region to its faces, or name the face with BCSegment(boundary=...) "
        "(Issue #2472)."
    )


def mixed_bc_from_regions(
    geometry: SupportsRegionMarking,
    bc_config: dict[str, BCSegment],
    dimension: int | None = None,
) -> BoundaryConditions:
    """
    Create mixed boundary conditions from marked regions (Issue #596 Phase 2.5).

    Convenient factory for region-based BCs without manual region_name assignment.

    On a structured grid, a region that covers whole faces and no part of any other face becomes one
    segment per face, named by ``boundary``: every reader then resolves it with no geometry, which is
    how the solvers and the FDM operators read a BC (#2472). The only region, on every face, becomes
    one unrestricted (uniform) segment. A region spanning several faces gives segments named
    ``"<template name>[<face>]"``. Any other region keeps ``region_name``: a face-level reader refuses
    it, since one condition per face cannot represent it, and a per-point reader matches it point by
    point when given the geometry. Such a region may not be named like a face ("x_max", "top"), since
    a reader without the geometry would read the name as that face.

    Args:
        geometry: Geometry with marked regions (must implement SupportsRegionMarking)
        bc_config: Mapping from region name to BC segment
            - Keys are region names from geometry.mark_region()
            - "default" key specifies fallback BC for unmarked regions
        dimension: Spatial dimension (inferred from geometry if None)

    Returns:
        BoundaryConditions object with region-based segments

    Raises:
        TypeError: If geometry doesn't implement SupportsRegionMarking
        ValueError: If region name in bc_config not found in geometry

    Example:
        >>> from mfgarchon.geometry import TensorProductGrid
        >>> from mfgarchon.geometry.boundary import BCSegment, BCType, mixed_bc_from_regions, no_flux_bc
        >>>
        >>> # Setup geometry with marked regions
        >>> geometry = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx_points=[50, 50], boundary_conditions=no_flux_bc(dimension=2))
        >>> geometry.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.1)
        >>> geometry.mark_region("outlet", boundary="x_max")
        >>>
        >>> # Define BCs via dictionary (no manual region_name assignment)
        >>> bc_config = {
        ...     "inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0),
        ...     "outlet": BCSegment(name="outlet_bc", bc_type=BCType.NEUMANN, value=0.0),
        ...     "default": BCSegment(name="default_bc", bc_type=BCType.PERIODIC)
        ... }
        >>>
        >>> # Create boundary conditions
        >>> bc = mixed_bc_from_regions(geometry, bc_config)
        >>> assert len(bc.segments) == 2  # inlet + outlet
        >>> assert bc.segments[0].region_name == "inlet"  # all of x_min, but part of y_min and y_max
        >>> assert bc.segments[1].boundary == "x_max"  # a whole face, resolved to it
        >>> assert bc.default_bc == BCType.PERIODIC
    """
    # Validate geometry supports region marking
    if not isinstance(geometry, SupportsRegionMarking):
        raise TypeError(
            f"mixed_bc_from_regions requires geometry implementing SupportsRegionMarking, got {type(geometry).__name__}"
        )

    # Infer dimension from geometry if not provided
    if dimension is None:
        dimension = geometry.dimension

    # Separate default BC from region-specific BCs, on a copy: popping from the caller's dict removed
    # their "default" entry.
    bc_config_copy = dict(bc_config)
    default_segment = bc_config_copy.pop("default", None)

    # Create segments with region_name field populated
    segments = []
    for region_name, segment_template in bc_config_copy.items():
        # Verify region exists in geometry
        available_regions = geometry.get_region_names()
        if region_name not in available_regions:
            raise ValueError(f"Region '{region_name}' not found in geometry. Available regions: {available_regions}")

        # Built first as a region-named segment, so BCSegment's own validation refuses a template that
        # already carries boundary / region / sdf_region / normal_direction, as it did before #2472.
        region_segment = replace(segment_template, region_name=region_name)

        # A whole-face region becomes its faces (#2472); anything else keeps region_name.
        mask = _region_mask_on_grid(geometry, region_name)
        covered, partial = _region_faces(mask) if mask is not None else (set(), set())
        if not (covered and not partial):
            label = parse_boundary_face(region_name)
            if label is not None:
                raise ValueError(
                    f"Region {region_name!r} is not a set of whole faces, and its name reads as a face "
                    f"label ({label.to_string()}). A reader without the geometry would read the name as "
                    "that face and apply the condition to all of it, or to no face if the grid has none "
                    "by that name. Rename the region (Issue #2472)."
                )
            segments.append(region_segment)
            continue
        faces = sorted(covered, key=lambda f: (f.axis, f.side))
        if len(bc_config_copy) == 1 and len(faces) == 2 * mask.ndim:
            # The only region, on every face: that is a uniform BC, as before #2472. Splitting it
            # made it mixed, which FP-FVM refuses and which runs a periodic BC through the per-face
            # path (review 3).
            segments.append(replace(region_segment, region_name=None))
            continue
        if len(faces) > 1 and segment_template.flux_capacity is not None:
            raise ValueError(
                f"Region {region_name!r} covers {len(faces)} faces and its segment carries a flux_capacity. "
                "One segment per face would count that capacity once per face. Give each face its own "
                "segment and capacity (Issue #2472)."
            )
        for face in faces:
            name = segment_template.name if len(faces) == 1 else f"{segment_template.name}[{face.to_string()}]"
            segments.append(replace(region_segment, region_name=None, boundary=face.to_string(), name=name))

    # Extract domain bounds from geometry if available
    # Use getattr pattern per CLAUDE.md (no hasattr for optional attributes)
    bounds = getattr(geometry, "bounds", None)
    domain_bounds = np.array(bounds) if bounds is not None else None

    # Create BoundaryConditions object.
    # Issue #1100: when no explicit "default" segment is supplied, fall back to
    # NO_FLUX (safe, mass-conserving, non-surprising) rather than the historical
    # PERIODIC. PERIODIC is applied only when the caller passes a default segment
    # whose bc_type is PERIODIC (the explicit path below).
    return BoundaryConditions(
        dimension=dimension,
        segments=segments,
        default_bc=default_segment.bc_type if default_segment else BCType.NO_FLUX,
        default_value=default_segment.value if default_segment else 0.0,
        domain_bounds=domain_bounds,
    )


# =============================================================================
# Backward Compatibility
# =============================================================================

# Alias for backward compatibility with code using MixedBoundaryConditions
MixedBoundaryConditions = BoundaryConditions
