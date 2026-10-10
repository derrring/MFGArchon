"""
Geometry trait enums, and the one protocol a dispatcher reads.

Three of the eight atomic traits of SPEC-GEO-1.0 (GEOMETRY_AND_TOPOLOGY.md), as enums that the
Geometry subclasses return from properties of the same name:

    ConnectivityType -> ``connectivity_type``: how neighbor relationships are determined
    StructureType    -> ``structure_type``:    whether the geometry has logical (i,j,k) indexing
    BoundaryDef      -> ``boundary_def``:      how the domain boundary is defined

Only ``boundary_def`` is read in the package: ``_has_implicit_boundary`` in
``geometry/boundary/dispatch.py`` checks ``isinstance(geometry, BoundaryAware)`` and compares it
with ``BoundaryDef.IMPLICIT``. ``connectivity_type`` and ``structure_type`` are declared and read
by nothing; whether they become the dispatch mechanism is the dispatch consolidation's decision.

References:
    - GEOMETRY_AND_TOPOLOGY.md (SPEC-GEO-1.0) Sections 2.1, 2.2, 2.5
    - Issue #732 Tier 1b
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

# =============================================================================
# Trait Enums
# =============================================================================


class ConnectivityType(Enum):
    """How neighbor relationships are determined.

    Declared by the Geometry subclasses' ``connectivity_type``; nothing in the package reads it.

    Values:
        IMPLICIT: Neighbors via stride arithmetic (TensorProductGrid).
            Zero memory overhead, favorable for SIMD/vectorization.
        EXPLICIT: Neighbors stored in adjacency structure (Mesh, NetworkGeometry).
            Memory bandwidth bound, supports arbitrary topology.
        DYNAMIC: Neighbors via runtime spatial search (ImplicitDomain, meshfree).
            High compute cost, suitable for meshfree/SPH methods.
    """

    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    DYNAMIC = "dynamic"


class StructureType(Enum):
    """Whether the geometry has regular logical indexing.

    Declared by the Geometry subclasses' ``structure_type``; nothing in the package reads it.

    Values:
        STRUCTURED: Nodes form a regular lattice with logical (i,j,k) coords.
        UNSTRUCTURED: Arbitrary point cloud, no concept of logical coords.
    """

    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"


class BoundaryDef(Enum):
    """How the domain boundary is defined.

    Influences BC enforcement strategy and boundary detection algorithms.

    Values:
        BOX: Axis-aligned hyper-rectangular bounds (AABB).
            Cheapest: boundary detection via coordinate comparison.
        MESH: Explicit boundary elements (facets from Gmsh).
            Supports curved and complex boundaries.
        IMPLICIT: Signed distance function phi(x) = 0.
            Dimension-agnostic, natural for CSG.
        NONE: No boundary (graphs, open domains). A periodic TensorProductGrid reports BOX.
    """

    BOX = "box"
    MESH = "mesh"
    IMPLICIT = "implicit"
    NONE = "none"


# =============================================================================
# Trait Protocol
# =============================================================================


@runtime_checkable
class BoundaryAware(Protocol):
    """Geometry that declares its boundary definition type.

    Read by ``get_applicator_for_geometry`` for MESHFREE discretisation, through
    ``_has_implicit_boundary``: ``BoundaryDef.IMPLICIT`` selects ``ImplicitApplicator``, and every
    other value ``MeshfreeApplicator``. No other value is dispatched on.
    """

    @property
    def boundary_def(self) -> BoundaryDef: ...
