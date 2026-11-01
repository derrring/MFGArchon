"""
Graph/Network boundary condition applicator for discrete MFG problems.

This module provides BC application for graph-based discretizations:
- Network MFG (GridNetwork, RandomNetwork, ScaleFreeNetwork)
- Maze MFG (MazeGeometry, VoronoiMaze, etc.)
- General graph domains (GraphGeometry)

BC Types Supported:
- Dirichlet: Fix values at boundary nodes
- Absorbing: Remove/zero density at exit nodes
- Source: Inject density at source nodes
- Custom: User-defined node-by-node BC

Design Philosophy:
    GraphApplicator provides a user-friendly unified interface for boundary
    conditions on discrete domains. It wraps direct node/edge access for
    convenience while exposing the low-level API for advanced users.

    Two approaches supported:
    1. High-level API: Specify BC by region names ("exit", "source", "boundary")
    2. Low-level API: Direct node indices and values

Created: 2025-11-27
Part of: Unified Boundary Condition Architecture
"""

from __future__ import annotations

import math
import numbers
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal, TypeGuard, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from mfgarchon.types.callable_protocols import NODE_RULE_SLOTS, NODE_VALUE_SLOTS, BoundCallable, bind_by_name

# Import base class for inheritance
from mfgarchon.utils.deprecation import NEVER_READ, retired_parameters

from .applicator_base import BaseGraphApplicator

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GraphBCType(Enum):
    """Boundary condition types for graph domains."""

    DIRICHLET = "dirichlet"  # Fixed value at nodes
    ABSORBING = "absorbing"  # Remove/zero density
    SOURCE = "source"  # Inject density
    NEUMANN = "neumann"  # Zero flux (no change)
    CUSTOM = "custom"  # User-defined function


#: Why an edge condition is refused: the one sentence `EdgeBC`, `GraphBCConfig` and `add_edge_bc` give.
_EDGES_NOT_APPLIED = (
    "an edge condition is not applied: no graph solver has an edge field it could act on, so it would be "
    "dropped. It is refused until an edge operator is ruled (#2493). For a condition at a node, use NodeBC."
)


def _refuse_a_foreign_bc_type(kind: str, name: str, bc_type: object) -> None:
    """A graph condition's type is a ``GraphBCType``. A continuum ``BCType`` shares two value strings with it,
    "dirichlet" and "neumann", while being a different member, and any other type matches no branch of
    ``GraphApplicator.apply``: either way the condition would be dropped without a word (#2292)."""
    if isinstance(bc_type, GraphBCType):
        return
    from .types import BCType

    hint = ""
    if isinstance(bc_type, BCType):
        hint = " That is the continuum alphabet, BCType"
        hint += f"; the graph one is GraphBCType.{bc_type.name}." if bc_type.name in GraphBCType.__members__ else "."
    raise TypeError(f"{kind} {name!r}: bc_type must be a GraphBCType, got {bc_type!r}.{hint} (#2292)")


# =============================================================================
# The forms a graph boundary-condition input is taken in (#2292): one table, read by every acceptor
# =============================================================================

_NODES = "a list, tuple or range of nodes, or a 1-D integer numpy array (an ndarray, not a subclass)"
#: The float dtypes a values array is taken in: each holds only numbers a float holds, so none saturates under the cast
#: to the float field. A long double is not among them, whatever its width on the host (#2292 review 5, M1).
_FLOATS = (np.float16, np.float32, np.float64)
_RULE = "a callable that takes the graph's node count and returns such nodes"
#: Each class of graph BC input and the forms it is taken in. Every acceptor reads its input through the reader of its
#: class below and coerces nothing on its own; any other form is refused by name, in these words.
ACCEPTED_FORMS: MappingProxyType[str, str] = MappingProxyType(
    {
        "node": "an int (a numbers.Integral that is not a bool) from 0 to the graph's last node",
        "nodes": _NODES,
        "condition_nodes": f"nodes ({_NODES}), or {_RULE}",
        "factory_nodes": (
            f"None, nodes ({_NODES}), or {_RULE}; None or an empty collection is no condition, except that "
            "create_maze_applicator detects its exits for exit_nodes=None while auto_detect_exits is True"
        ),
        "value": (
            "a real number a float holds (a numbers.Real that is not a bool, which float() converts, to a finite float "
            "if it is finite), or a function of (t, node), time first, whose parameters are named after them: bound once "
            "and called by name, and refused otherwise; one that reads t is read only where a time is given, and not "
            "into a 2-D (Nt, N) field"
        ),
        "rule": (
            "a function of (t, node, field), time first, bound as a value is; field is the node values the CUSTOM "
            "condition is applied to"
        ),
        "values": (
            "a value, or a list, tuple or 1-D real numpy array (an ndarray, not a subclass, of an integer dtype or of "
            "float16, float32 or float64) of real numbers a float holds, one per node"
        ),
        "conditions": "a list or tuple",
        "threshold": "a finite real number (a numbers.Real that is not a bool), compared as given",
    }
)


def _shown(item: object) -> str:
    """How a refusal names what it was given."""
    if item is None:
        return "None"
    if isinstance(item, np.ndarray):
        # A subclass is named as itself, so a refused memmap or masked array is not described as the array taken; so
        # is a long double, which prints as float64 where it is one.
        dtype = "longdouble" if item.dtype.type is np.longdouble else item.dtype
        return f"a {item.ndim}-D {dtype} {'array' if type(item) is np.ndarray else type(item).__name__}"
    name = type(item).__name__
    return f"{'an' if name[0].lower() in 'aeiou' else 'a'} {name}"


def _is_real(item: object) -> bool:
    return isinstance(item, numbers.Real) and not isinstance(item, (bool, np.bool_))


def _plain_array(item: object) -> TypeGuard[np.ndarray]:
    """A numpy array of the base class, the one whose ``tolist()`` the readers' meaning rests on. A subclass can
    redefine it: a masked array's masked entries read as None, which numpy wrote as NaN (#2292 review 4, F1)."""
    return type(item) is np.ndarray


def _is_value(item: object) -> bool:
    """A real number a float holds: a value is written into a float field, so ``float()`` must succeed and keep it. It
    raises for an int beyond float range or a Real that defines no conversion, and saturates to inf for some Reals
    beyond it (sympy's, mpmath's, gmpy2's mpfr), so a finite number that becomes inf, or a number that becomes NaN, is
    refused too (#2292 reviews 4 and 5)."""
    if not _is_real(item):
        return False
    try:
        as_float = float(cast("float", item))
    except (ArithmeticError, TypeError, ValueError):
        # The conversion's own failures. A Real whose __float__ raises anything else propagates that error as itself:
        # the fail-fast ratchet admits no broad handler (#2292 review 5, SF3).
        return False
    if math.isnan(as_float):
        return item != item  # only a NaN is a NaN
    return not math.isinf(as_float) or item == as_float


def _nodes(items: object, where: str, form: str = "nodes") -> tuple:
    """``items`` in a form ``ACCEPTED_FORMS["nodes"]`` names, as a tuple of its entries, or refused by name in the words
    of ``ACCEPTED_FORMS[form]``, the acceptor's own entry. The entries are checked against a graph where an applicator
    takes them (`GraphApplicator._checked_nodes`)."""
    if isinstance(items, (list, tuple, range)):
        return tuple(items)
    # A masked array is not one: its masked entries read as None, so the nodes it declares are not its entries.
    if _plain_array(items) and items.ndim == 1 and items.dtype.kind in "iu":
        return tuple(items.tolist())
    raise TypeError(f"{where} is {_shown(items)}; it takes {ACCEPTED_FORMS[form]} (#2292).")


def _conditions(items: object, where: str, of: str) -> tuple:
    """``items`` in a form ``ACCEPTED_FORMS["conditions"]`` names, as a tuple, or refused by name."""
    if isinstance(items, (list, tuple)):
        return tuple(items)
    raise TypeError(f"{where} is {_shown(items)}; it takes {ACCEPTED_FORMS['conditions']} of {of} (#2292).")


class _Constant:
    """A constant node value, read as a bound callable is, so every reader makes one call (#2296)."""

    __slots__ = ("constant",)
    passes: tuple[str, ...] = ()

    def __init__(self, constant: float):
        self.constant = constant

    def __call__(self, **_slots: object) -> float:
        return self.constant


# The wraps a refusal advises, for each order a callable may be written in: one written in the order before #2378
# phase 5 and wrapped time first would be read swapped.
_VALUE_WRAP = "`lambda t, node: f(t, node)`, or `lambda t, node: f(node, t)` for one that takes the node first"
_RULE_WRAP = (
    "`lambda t, node, field: f(t, node, field)`, or `lambda t, node, field: f(field, node, t)` for one written in "
    "`set_custom_bc`'s old order"
)


def _value_reader(value: object, where: str) -> BoundCallable | _Constant:
    """``value`` in the form ``ACCEPTED_FORMS["value"]`` names, as the one object every reader calls, or refused by
    name. A callable is bound here, once, to ``(t, node)`` by its parameter names, by the rule a grid boundary value is
    bound by (`bind_by_name`, #2296): one written in the order before #2378 phase 5, ``(node, t)``, is refused rather
    than swapped, and so is one whose parameters cannot be read, which position alone would have swapped."""
    if callable(value):
        try:
            return bind_by_name(
                value,
                NODE_VALUE_SLOTS,
                role="graph node value",
                ruling="#2296",
                wrap=_VALUE_WRAP,
                wrap_unnamed=_VALUE_WRAP,
            )
        except TypeError as refused:
            raise TypeError(f"{where}: {refused}") from refused
    if not _is_value(value):
        shown = f"{_shown(value)}, a real number no float holds" if _is_real(value) else _shown(value)
        raise TypeError(f"{where} is {shown}; it takes {ACCEPTED_FORMS['value']} (#2292).")
    return _Constant(float(cast("float", value)))


def _rule_reader(rule: object, where: str) -> BoundCallable:
    """A CUSTOM condition's ``rule`` in the form ``ACCEPTED_FORMS["rule"]`` names, bound to ``(t, node, field)``, or
    refused by name (#2296)."""
    if not callable(rule):
        raise TypeError(f"{where} is {_shown(rule)}; it takes {ACCEPTED_FORMS['rule']} (#2296).")
    try:
        return bind_by_name(
            rule, NODE_RULE_SLOTS, role="graph CUSTOM rule", ruling="#2296", wrap=_RULE_WRAP, wrap_unnamed=_RULE_WRAP
        )
    except TypeError as refused:
        raise TypeError(f"{where}: {refused}") from refused


def _graph_value_at(
    read: BoundCallable | _Constant, where: str, *, t: float | None, node: int, field: NDArray | None = None
) -> float:
    """The one place a graph node's value or CUSTOM rule is read (#2296). One that reads ``t`` is refused by name when
    no time is given: a default of 0 read it at t = 0 in silence."""
    if t is None and "t" in read.passes:
        raise TypeError(f"{where}: it reads t, and no time was given. Pass t where the condition is applied (#2296).")
    slots: dict[str, object] = {"t": t, "node": node} if field is None else {"t": t, "node": node, "field": field}
    value = read(**slots)
    try:
        return float(value)
    except (TypeError, ValueError) as refused:
        raise TypeError(
            f"{where}: it returned {_shown(value)}, not one real number; a node's value is one (#2296)."
        ) from refused


def _refuse_a_time_read_into_rows(read: BoundCallable | _Constant, where: str) -> None:
    """A 2-D ``(Nt, N)`` field takes one value per node for every row, at the one time it is applied at, so a value or
    rule that reads ``t`` is refused there by name rather than written at that time into every row (#2296)."""
    if "t" in read.passes:
        raise TypeError(
            f"{where}: it reads t, and a 2-D (Nt, N) field would take its value at one time in every row. Apply the "
            "condition row by row, each at its own time (#2296)."
        )


def _node_values(values: object, count: int, where: str) -> tuple | None:
    """``values`` in a form ``ACCEPTED_FORMS["values"]`` names: None for a value, which every node takes, else one
    value per node, or refused by name."""
    if callable(values) or _is_value(values):
        return None
    if isinstance(values, (list, tuple)) and all(_is_value(v) for v in values):
        per_node = tuple(values)
    elif _plain_array(values) and values.ndim == 1 and (values.dtype.kind in "iu" or values.dtype.type in _FLOATS):
        per_node = tuple(values.tolist())
    else:
        shown = f"{_shown(values)}, a real number no float holds" if _is_real(values) else _shown(values)
        raise TypeError(
            f"{where}: values is {shown}; it takes {ACCEPTED_FORMS['values']}, a value being "
            f"{ACCEPTED_FORMS['value']} (#2292)."
        )
    if len(per_node) != count:
        raise ValueError(
            f"{where}: values holds {len(per_node)} value(s) for {count} node(s); it takes {ACCEPTED_FORMS['values']} "
            "(#2292)."
        )
    return per_node


def _threshold(threshold: object, where: str) -> float:
    """``threshold`` in the form ``ACCEPTED_FORMS["threshold"]`` names, unconverted, or refused by name. Finite by
    comparison: float() would round a Fraction, and math.isfinite overflows on a large int (#2292)."""
    if not _is_real(threshold) or not -math.inf < cast("float", threshold) < math.inf:
        raise TypeError(f"{where} is {threshold!r}; it takes {ACCEPTED_FORMS['threshold']} (#2292).")
    return cast("float", threshold)  # the cast is for mypy: the value is returned as given


def _given_nodes(nodes: object, where: str, form: str = "factory_nodes") -> tuple | Callable[[int], list[int]] | None:
    """A node argument in the form ``ACCEPTED_FORMS[form]`` names: a callable as given, None for none where the form
    takes it, or nodes read by `_nodes`. An empty collection is no condition (#2292)."""
    if callable(nodes) or (nodes is None and form == "factory_nodes"):
        return cast("Callable[[int], list[int]] | None", nodes)
    return _nodes(nodes, where, form)


@dataclass(frozen=True)
class NodeBC:
    """
    Boundary condition specification for a set of nodes.

    Frozen, with its nodes kept as a tuple, so a condition cannot change after it is built; its nodes are checked
    against a graph when an applicator takes it (#2292).

    Attributes:
        nodes: Node indices, kept as a tuple, or callable(num_nodes) -> indices
        bc_type: Type of boundary condition, a ``GraphBCType``
        value: BC value, a constant or a callable of (t, node), time first (#2296)
        name: Optional name for this BC region (e.g., "exit", "source")
        rule: A CUSTOM condition's rule, a callable of (t, node, field) -> the node's new value, and only CUSTOM's
    """

    nodes: Sequence[int] | Callable[[int], list[int]]
    bc_type: GraphBCType = GraphBCType.DIRICHLET
    value: float | Callable[..., float] = 0.0
    name: str = "boundary"
    rule: Callable[..., float] | None = None

    def __post_init__(self) -> None:
        _refuse_a_foreign_bc_type("NodeBC", self.name, self.bc_type)
        if not callable(self.nodes):
            object.__setattr__(self, "nodes", _nodes(self.nodes, f"NodeBC {self.name!r}: nodes", "condition_nodes"))
        # A CUSTOM condition's rule reads the field as well, so it has a field of its own, and `value` is not read for
        # it: one field carrying both contracts applied, raised or did nothing by the caller's signature (#2296).
        if self.bc_type is GraphBCType.CUSTOM:
            if self.rule is None or not (_is_value(self.value) and self.value == 0):
                raise TypeError(
                    f"NodeBC {self.name!r}: a CUSTOM condition takes its rule as rule=, {ACCEPTED_FORMS['rule']}, and "
                    "no value; a value is not read for it (#2296)."
                )
            object.__setattr__(self, "_read", _rule_reader(self.rule, f"NodeBC {self.name!r}: rule"))
        else:
            if self.rule is not None:
                raise TypeError(
                    f"NodeBC {self.name!r}: rule= is a CUSTOM condition's; a {self.bc_type.name} condition takes value "
                    "(#2296)."
                )
            object.__setattr__(self, "_read", _value_reader(self.value, f"NodeBC {self.name!r}: value"))
        # NEUMANN on a graph is zero flux, the graph Laplacian's own condition at a node, and
        # `GraphApplicator.apply` leaves both fields unchanged for it. A non-zero flux would be
        # dropped without a word, so it is refused (#1948).
        if self.bc_type == GraphBCType.NEUMANN and (callable(self.value) or self.value != 0):
            raise NotImplementedError(
                f"NodeBC {self.name!r}: GraphBCType.NEUMANN is zero flux, and a non-zero flux "
                f"(value={self.value!r}) is not implemented on graphs. It would be dropped rather than "
                "imposed. Use value=0.0 (the default), or a SOURCE node to inject mass (#1948)."
            )

    def get_nodes(self, num_nodes: int) -> list[int]:
        """Get node indices, resolving callable if needed."""
        if callable(self.nodes):
            return list(_nodes(self.nodes(num_nodes), f"NodeBC {self.name!r}: nodes({num_nodes})"))
        return list(self.nodes)

    @property
    def _reader(self) -> BoundCallable | _Constant:
        """The value's or rule's binding, made once when this condition was built (#2296)."""
        return cast("BoundCallable | _Constant", self.__dict__["_read"])

    def get_value(self, node: int, t: float | None) -> float:
        """This condition's value at ``node`` and time ``t``, through the binding made when it was built (#2296)."""
        if self.bc_type is GraphBCType.CUSTOM:
            raise TypeError(f"NodeBC {self.name!r}: a CUSTOM condition has a rule, read with the field, not a value.")
        return _graph_value_at(self._reader, f"NodeBC {self.name!r}: value", t=t, node=node)


@dataclass(frozen=True)
class EdgeBC:
    """
    Boundary condition specification for edges (flow-based problems).

    Refused when built: no graph solver applies an edge condition (#2493).

    Attributes:
        edges: List of (from_node, to_node) tuples or callable
        bc_type: Type of boundary condition
        value: Edge flow value or callable(from_node, to_node, t) -> value
        name: Optional name for this BC region
    """

    edges: list[tuple[int, int]] | Callable[[int], list[tuple[int, int]]]
    bc_type: GraphBCType = GraphBCType.DIRICHLET
    value: float | Callable[[int, int, float], float] = 0.0
    name: str = "edge_boundary"

    def __post_init__(self) -> None:
        raise NotImplementedError(f"EdgeBC {self.name!r}: {_EDGES_NOT_APPLIED}")


@dataclass(frozen=True)
class GraphBCConfig:
    """
    Configuration for graph boundary conditions.

    Provides a declarative way to specify BCs for graph problems. Frozen, with its conditions kept as tuples,
    so a configuration checked once cannot change after it is handed over (#2292).

    Example:
        >>> config = GraphBCConfig(
        ...     node_bcs=[
        ...         NodeBC(nodes=[0, 1], bc_type=GraphBCType.DIRICHLET, value=0.0, name="exit"),
        ...         NodeBC(nodes=[10, 11], bc_type=GraphBCType.SOURCE, value=1.0, name="source"),
        ...     ],
        ... )
        >>> applicator = GraphApplicator.from_config(config, num_nodes=20)
    """

    node_bcs: tuple[NodeBC, ...] = ()
    edge_bcs: tuple[EdgeBC, ...] = ()
    default_bc_type: GraphBCType = NEVER_READ  # refused when passed; never read (#2648)

    def __post_init__(self) -> None:
        # The tuple first: an iterator is consumed by the check below, and would be stored empty (#2292).
        object.__setattr__(self, "node_bcs", _conditions(self.node_bcs, "GraphBCConfig: node_bcs", "NodeBC"))
        object.__setattr__(self, "edge_bcs", _conditions(self.edge_bcs, "GraphBCConfig: edge_bcs", "EdgeBC"))
        for i, bc in enumerate(self.node_bcs):
            if not isinstance(bc, NodeBC):
                raise TypeError(
                    f"GraphBCConfig: node_bcs[{i}] is {_shown(bc)}, not a NodeBC; node_bcs takes "
                    f"{ACCEPTED_FORMS['conditions']} of NodeBC (#2292)."
                )
        if self.edge_bcs:
            raise NotImplementedError(f"GraphBCConfig: edge_bcs is not empty, and {_EDGES_NOT_APPLIED}")


GraphBCConfig.__init__ = retired_parameters(  # type: ignore[method-assign]
    {
        "default_bc_type": "a node without a NodeBC gets no condition; give each node that needs one a NodeBC (#2493)",
    },
    issue="#2648",
    since="v0.22.0",
    never_read=True,
)(GraphBCConfig.__init__)


class GraphApplicator(BaseGraphApplicator):
    """
    Boundary condition applicator for graph/network/maze domains.

    Provides both high-level interface (named regions, automatic detection)
    and low-level interface (direct node/edge specification).

    Inherits from BaseGraphApplicator for consistent interface across
    all BC applicators.

    Works with:
    - NetworkMFGProblem (GridNetwork, RandomNetwork, ScaleFreeNetwork)
    - Maze generators (MazeGeometry, VoronoiMaze, etc.)
    - General graph geometries (GraphGeometry)

    Examples:
        High-level API (recommended for most users):

        >>> # From network geometry
        >>> network = GridNetwork(width=10, height=10)
        >>> applicator = GraphApplicator(num_nodes=network.num_nodes)
        >>> applicator.set_dirichlet_nodes([0, 99], value=0.0, name="exit")
        >>> applicator.set_source_nodes([50], value=1.0)
        >>> u_bc = applicator.apply(u, t=0.0)

        >>> # From configuration
        >>> config = GraphBCConfig(node_bcs=[
        ...     NodeBC(nodes=[0], bc_type=GraphBCType.DIRICHLET, value=0.0)
        ... ])
        >>> applicator = GraphApplicator.from_config(config, num_nodes=100)

        Low-level API (for advanced users):

        >>> # Direct node-value specification
        >>> applicator = GraphApplicator(num_nodes=100)
        >>> u_bc = applicator.apply_direct(u, boundary_nodes=[0, 99], values=[0.0, 0.0])

    """

    #: (type, field) cells `apply` leaves unchanged on purpose, each with its reason. Declared rather than
    #: left to a source comment, so a caller and the conformance table can tell a deliberate no-op
    #: from an unhandled cell (#1948, maintainer ruling 2026-10-04).
    _DECLARED_IDENTITY: ClassVar[MappingProxyType[tuple[GraphBCType, str], str]] = MappingProxyType(
        {
            (GraphBCType.DIRICHLET, "density"): (
                "a Dirichlet pin is a condition on the value u (#1471); the exit that removes density is ABSORBING (#1478)"
            ),
            (GraphBCType.SOURCE, "value"): "a source injects density; u has no source term at the node",
            (GraphBCType.NEUMANN, "value"): "zero flux is the graph Laplacian's own condition at a node",
            (GraphBCType.NEUMANN, "density"): "zero flux is the graph Laplacian's own condition at a node",
        }
    )

    #: The fields each condition writes at its nodes in `apply` (#2292): every field `_DECLARED_IDENTITY` does not
    #: declare it leaves unchanged, derived below the class. Two conditions that write one field at one node would leave
    #: it to whichever `apply` reads last, so `add_node_bc` refuses the second.
    _WRITES: ClassVar[MappingProxyType[GraphBCType, tuple[str, ...]]]

    def __init__(self, num_nodes: int):
        """
        Initialize graph BC applicator.

        Args:
            num_nodes: Total number of nodes in the graph
        """
        super().__init__(num_nodes)

        # Node BC storage; a name may name several conditions, and a lookup by it refuses that (#2292)
        self._node_bcs: list[NodeBC] = []
        self._node_bcs_by_name: dict[str, list[NodeBC]] = {}

        # Cache for resolved node indices
        self._resolved_boundary_nodes: set[int] | None = None
        self._resolved_source_nodes: set[int] | None = None

    @classmethod
    def from_config(cls, config: GraphBCConfig, num_nodes: int) -> GraphApplicator:
        """
        Create applicator from configuration.

        Args:
            config: GraphBCConfig specification
            num_nodes: Total number of nodes

        Returns:
            Configured GraphApplicator
        """
        if not isinstance(config, GraphBCConfig):
            raise TypeError(
                f"GraphApplicator.from_config: config is a {type(config).__name__}, not a GraphBCConfig (#2292)."
            )
        applicator = cls(num_nodes)
        for node_bc in config.node_bcs:
            applicator.add_node_bc(node_bc)
        return applicator

    @classmethod
    def from_network_geometry(
        cls,
        geometry,
        boundary_type: Literal["leaf", "degree", "spatial", "custom"] = "leaf",
        **kwargs,
    ) -> GraphApplicator:
        """
        Create applicator from network geometry with automatic boundary detection.

        Args:
            geometry: BaseNetworkGeometry or GraphGeometry instance
            boundary_type: How to detect boundary nodes
                - "leaf": Nodes with degree 1 (dead ends)
                - "degree": Nodes with degree <= threshold
                - "spatial": Nodes on spatial bounding box (for embedded networks)
                - "custom": Use provided node list
            **kwargs: Additional arguments
                - degree_threshold: For "degree" type (default: 1)
                - boundary_nodes: For "custom" type, which needs it
                - boundary_value: The boundary nodes' Dirichlet value, for every type (default: 0.0)
                Any other keyword is refused (#2292).

        Returns:
            Configured GraphApplicator

        Example:
            >>> network = GridNetwork(10, 10)
            >>> applicator = GraphApplicator.from_network_geometry(network, boundary_type="leaf")
        """
        read = {"leaf": set(), "degree": {"degree_threshold"}, "spatial": set(), "custom": {"boundary_nodes"}}
        if boundary_type not in read:
            raise ValueError(f"Unknown boundary_type: {boundary_type}")
        unread = sorted(set(kwargs) - read[boundary_type] - {"boundary_value"})
        if unread:
            raise TypeError(
                f"GraphApplicator.from_network_geometry: boundary_type={boundary_type!r} reads "
                f"{sorted(read[boundary_type] | {'boundary_value'})}, and not {unread} (#2292)."
            )
        _value_reader(kwargs.get("boundary_value", 0.0), "GraphApplicator.from_network_geometry: boundary_value")
        num_nodes = _num_nodes(geometry)

        applicator = cls(num_nodes)
        boundary_nodes: Sequence[int] | Callable[[int], list[int]] | None

        # Detect boundary nodes based on type
        if boundary_type == "leaf":
            boundary_nodes = cls._detect_leaf_nodes(geometry)
        elif boundary_type == "degree":
            threshold = _threshold(
                kwargs.get("degree_threshold", 1), "GraphApplicator.from_network_geometry: degree_threshold"
            )
            boundary_nodes = cls._detect_low_degree_nodes(geometry, threshold)
        elif boundary_type == "spatial":
            boundary_nodes = cls._detect_spatial_boundary_nodes(geometry)
        else:  # "custom"
            boundary_nodes = _given_nodes(
                kwargs.get("boundary_nodes"), "GraphApplicator.from_network_geometry: boundary_nodes", "condition_nodes"
            )

        # A detector's list, or the given nodes read once into a tuple: empty is no condition. A truth test on the
        # given object dropped a bare 0 and a one-element array of node 0 (#2292).
        if boundary_nodes:
            applicator.set_dirichlet_nodes(boundary_nodes, value=kwargs.get("boundary_value", 0.0), name="boundary")

        return applicator

    @staticmethod
    def _detect_leaf_nodes(geometry) -> list[int]:
        """Detect leaf nodes (degree 1) in the graph."""
        degrees = GraphApplicator._node_degrees(geometry)
        return [int(i) for i in np.where(degrees == 1)[0]]

    @staticmethod
    def _node_degrees(geometry) -> np.ndarray:
        """Combinatorial degree: how many neighbours each node has.

        `(adj != 0).sum(axis=1)`, not `adj.sum(axis=1)`. The latter is the weighted degree -- the
        node STRENGTH -- and on a weighted graph it is not the degree at all. Both detectors below
        used it, so on a path graph `0-1-2-3` with edge weight 0.5 they returned `[1, 2]`, the two
        NON-leaves, and the boundary condition was applied to the interior. #1940

        ~~The declared owner, `network_backend.node_degrees`, returns the combinatorial degree, so
        this agrees with it.~~ [CORRECTED 2026-08-15] The owner does not agree with ITSELF: of the
        three backends, igraph (`:186`) and networkit (`:271`) return the combinatorial degree while
        networkx (`:362`) returns `dict(graph.degree(weight="weight"))`, the strength. Measured on
        the weighted path above: igraph gives [1 2 2 1], networkx gives [0.5 1. 1. 0.5]. So there is
        no single owner to route through -- that is a pre-existing single-source violation, filed
        separately. This computes the quantity a leaf detector needs and says which one it is.

        The weighted sum is NOT wrong everywhere, and the correct sites are more than I first listed:
        `network_geometry.py:174`, `network_backend.py:161/243`, `alg/numerical/coupling/
        graph_coupling.py:238` (another `D - A`), and `alg/numerical/network_solvers/
        fp_network.py:240`, whose CFL bound needs it -- measured, lambda_max of the weighted
        Laplacian is 12.55, so 2*max_weighted_degree = 17.0 bounds it and 2*max_combinatorial = 6.0
        does not. All deliberately untouched. `network_geometry.py:217/218` are a third case:
        `metadata["average_degree"]` and `["max_degree"]` are strengths under a degree name --
        pre-existing, metadata only, not changed here.
        """
        # Issue #543: Use try/except instead of hasattr() for optional attribute
        try:
            network_data = geometry.network_data
        except AttributeError:
            network_data = None
        adj = None if network_data is None else network_data.adjacency_matrix
        if adj is None:
            # A graph geometry without `network_data`, such as a maze, gives its own adjacency (#2292).
            try:
                adj = geometry.get_adjacency_matrix()
            except AttributeError:
                raise TypeError(
                    f"{type(geometry).__name__} gives neither network_data.adjacency_matrix nor get_adjacency_matrix(), "
                    "so no node degree can be read from it (#2292)."
                ) from None
        return np.asarray((adj != 0).sum(axis=1)).flatten()

    @staticmethod
    def _detect_low_degree_nodes(geometry, threshold: float) -> list[int]:
        """Detect nodes with degree <= threshold."""
        degrees = GraphApplicator._node_degrees(geometry)
        return [int(i) for i in np.where(degrees <= threshold)[0]]

    @staticmethod
    def _detect_spatial_boundary_nodes(geometry, tolerance: float = 1e-6) -> list[int]:
        """Detect nodes on spatial bounding box (for spatially-embedded networks)."""
        # Issue #543: Use try/except instead of hasattr() for optional attribute
        try:
            network_data = geometry.network_data
        except AttributeError:
            network_data = None
        positions = None if network_data is None else network_data.node_positions
        if positions is None:
            # A graph geometry without `network_data`, such as a maze, gives its own positions (#2292).
            try:
                positions = geometry.get_node_positions()
            except AttributeError:
                positions = None
        if positions is None:
            raise TypeError(
                f"{type(geometry).__name__} gives no node positions, so no node lies on its bounding box; "
                "use boundary_type='leaf', 'degree' or 'custom' (#2292)."
            )
        positions = np.asarray(positions)
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        return [
            i
            for i, pos in enumerate(positions)
            if np.any(np.abs(pos - min_coords) < tolerance) or np.any(np.abs(pos - max_coords) < tolerance)
        ]

    # =========================================================================
    # High-Level API: Named Region Specification
    # =========================================================================

    def set_dirichlet_nodes(
        self,
        nodes: Sequence[int] | Callable[[int], list[int]],
        value: float | Callable[..., float] = 0.0,
        name: str = "dirichlet",
    ) -> GraphApplicator:
        """
        Set Dirichlet BC on specified nodes.

        Args:
            nodes: Node indices or callable(num_nodes) -> indices
            value: BC value, a constant or a callable of (t, node), time first (#2296)
            name: Name for this BC region

        Returns:
            self (for method chaining)

        Example:
            >>> applicator.set_dirichlet_nodes([0, 99], value=0.0, name="exit")
            >>> applicator.set_dirichlet_nodes(
            ...     nodes=[10],
            ...     value=lambda t, node: np.sin(t),
            ...     name="oscillating"
            ... )
        """
        bc = NodeBC(nodes=nodes, bc_type=GraphBCType.DIRICHLET, value=value, name=name)
        return self.add_node_bc(bc)

    def set_absorbing_nodes(
        self,
        nodes: Sequence[int] | Callable[[int], list[int]],
        name: str = "absorbing",
    ) -> GraphApplicator:
        """
        Set absorbing BC on specified nodes (density goes to zero).

        Commonly used for exit nodes in crowd dynamics.

        Args:
            nodes: Node indices or callable
            name: Name for this BC region

        Returns:
            self (for method chaining)

        Example:
            >>> # Exit nodes where agents leave the system
            >>> applicator.set_absorbing_nodes([0, 99], name="exit")
        """
        bc = NodeBC(nodes=nodes, bc_type=GraphBCType.ABSORBING, value=0.0, name=name)
        return self.add_node_bc(bc)

    def set_source_nodes(
        self,
        nodes: Sequence[int] | Callable[[int], list[int]],
        value: float | Callable[..., float] = 1.0,
        name: str = "source",
    ) -> GraphApplicator:
        """
        Set source BC on specified nodes (density injection).

        Args:
            nodes: Node indices or callable
            value: Source strength, a constant or a callable of (t, node), time first (#2296)
            name: Name for this BC region

        Returns:
            self (for method chaining)

        Example:
            >>> # Entry points where agents enter the system
            >>> applicator.set_source_nodes([50], value=1.0, name="entry")
        """
        bc = NodeBC(nodes=nodes, bc_type=GraphBCType.SOURCE, value=value, name=name)
        return self.add_node_bc(bc)

    def set_custom_bc(
        self,
        nodes: Sequence[int] | Callable[[int], list[int]],
        bc_func: Callable[..., float],
        name: str = "custom",
    ) -> GraphApplicator:
        """
        Set custom BC with user-defined function.

        Args:
            nodes: Node indices or callable
            bc_func: The rule, a callable of (t, node, field) -> the node's new value, time first (#2296)
            name: Name for this BC region

        Returns:
            self (for method chaining)

        Example:
            >>> def robin_bc(t, node, field):
            ...     # Robin-like: u + alpha * du/dn = g
            ...     return 0.5 * field[node]
            >>> applicator.set_custom_bc([0, 99], robin_bc)
        """
        # Checked here first, so a refusal names bc_func as the caller gave it (#2292 review 4, S1).
        _rule_reader(bc_func, "GraphApplicator.set_custom_bc: bc_func")
        bc = NodeBC(nodes=nodes, bc_type=GraphBCType.CUSTOM, rule=bc_func, name=name)
        return self.add_node_bc(bc)

    def add_node_bc(self, bc: NodeBC) -> GraphApplicator:
        """Add a node BC specification. Its nodes are checked against this graph now (#2292)."""
        if not isinstance(bc, NodeBC):
            raise TypeError(f"GraphApplicator.add_node_bc: got a {type(bc).__name__}, not a NodeBC (#2292).")
        nodes = set(self._resolve_nodes(bc))
        for other in self._node_bcs:
            shared = sorted(nodes & set(self._resolve_nodes(other)))
            fields = sorted(set(self._WRITES[bc.bc_type]) & set(self._WRITES[other.bc_type]))
            if shared and fields:
                raise ValueError(
                    f"GraphApplicator.add_node_bc: NodeBC {bc.name!r} ({bc.bc_type.name}) and {other.name!r} "
                    f"({other.bc_type.name}) both write the {' and '.join(fields)} field at node(s) {shared}, so `apply` "
                    "would keep whichever it reads last, silently. Give each node one condition per field (#2292)."
                )
        self._node_bcs.append(bc)
        self._node_bcs_by_name.setdefault(bc.name, []).append(bc)
        self._invalidate_cache()
        return self

    def add_edge_bc(self, bc: EdgeBC) -> GraphApplicator:
        """Refused: no graph solver applies an edge condition (#2493)."""
        raise NotImplementedError(f"GraphApplicator.add_edge_bc: {_EDGES_NOT_APPLIED}")

    def _checked_nodes(self, nodes, where: str, param: str = "nodes") -> list[int]:
        """``nodes`` as a list of node indices of this graph, or refused by name: an index outside
        ``[0, num_nodes)`` would otherwise be skipped without a word (#2292)."""
        indices = list(_nodes(nodes, f"{where}: {param}"))
        foreign = [n for n in indices if isinstance(n, (bool, np.bool_)) or not isinstance(n, numbers.Integral)]
        if foreign:
            raise TypeError(
                f"{where}: node(s) {foreign!r} are not node indices; a node is {ACCEPTED_FORMS['node']} (#2292)."
            )
        indices = [int(n) for n in indices]
        outside = [n for n in indices if not 0 <= n < self.num_nodes]
        if outside:
            raise ValueError(
                f"{where}: node(s) {outside} are not nodes of this graph; a node is {ACCEPTED_FORMS['node']}, here "
                f"0..{self.num_nodes - 1} (#2292)."
            )
        return indices

    def _resolve_nodes(self, bc: NodeBC) -> list[int]:
        """The one reading of a condition's nodes, a callable resolved against this graph, each one checked."""
        return self._checked_nodes(bc.get_nodes(self.num_nodes), f"NodeBC {bc.name!r}")

    def _invalidate_cache(self) -> None:
        """Invalidate cached resolved indices."""
        self._resolved_boundary_nodes = None
        self._resolved_source_nodes = None

    # =========================================================================
    # Application Methods
    # =========================================================================

    def apply(
        self,
        field: NDArray[np.floating],
        t: float | None = None,
        field_type: Literal["value", "density"] = "value",
    ) -> NDArray[np.floating]:
        """
        Apply all configured boundary conditions to a field.

        Args:
            field: Node values (shape: (num_nodes,) or (Nt, num_nodes))
            t: Current time. A condition whose value reads t needs it, and is refused on a 2-D field (#2296).
            field_type: Type of field
                - "value": HJB value function (u)
                - "density": FP density (m)

        Returns:
            Field with BCs applied

        Example:
            >>> u = np.ones(100)
            >>> u_bc = applicator.apply(u, t=0.5, field_type="value")

        Raises:
            ValueError: If `field_type` is not "value" or "density".
        """
        # `Literal` is a type annotation, not a runtime check. Every branch below tests
        # `field_type == "value"` or `== "density"`, so any other string -- including "VALUE",
        # "hjb", or a typo -- matched nothing and the field came back unmodified with no error.
        # The failure mode was silence, which is what makes it worth a guard rather than a
        # docstring line. #1940
        if field_type not in ("value", "density"):
            raise ValueError(
                f"field_type must be 'value' or 'density', got {field_type!r}. "
                f"'value' is the HJB value function u; 'density' is the FP density m. "
                f"They receive different boundary conditions by adjoint duality, so there is no "
                f"safe default and an unrecognised name is not applied to either."
            )

        result = field.copy()
        is_2d = result.ndim == 2

        for bc in self._node_bcs:
            for node in self._resolve_nodes(bc):
                if bc.bc_type == GraphBCType.DIRICHLET:
                    # Issue #1471: a Dirichlet value pin belongs to the VALUE field (HJB u), not the
                    # density field (FP m) — pinning m to a value-function value is wrong.
                    if field_type == "value":
                        value = self._value_of(bc, node, t, is_2d)
                        if is_2d:
                            result[:, node] = value
                        else:
                            result[node] = value

                elif bc.bc_type == GraphBCType.ABSORBING:
                    # Issue #1478: an absorbing / exit node is ONE physical BC with dual operations
                    # (adjoint duality): the HJB value carries the exit cost (Dirichlet u = value), and
                    # the FP density is absorbed (m -> 0, mass exits). Apply the right one per field.
                    if field_type == "value":
                        value = self._value_of(bc, node, t, is_2d)
                        if is_2d:
                            result[:, node] = value
                        else:
                            result[node] = value
                    else:  # density
                        if is_2d:
                            result[:, node] = 0.0
                        else:
                            result[node] = 0.0

                elif bc.bc_type == GraphBCType.SOURCE:
                    if field_type == "density":
                        value = self._value_of(bc, node, t, is_2d)
                        if is_2d:
                            result[:, node] = value
                        else:
                            result[node] = value
                    # For value functions, source nodes don't modify u

                elif bc.bc_type == GraphBCType.NEUMANN:
                    # No change for Neumann (handled by solver)
                    pass

                elif bc.bc_type == GraphBCType.CUSTOM:
                    where = f"NodeBC {bc.name!r}: rule"
                    if is_2d:
                        _refuse_a_time_read_into_rows(bc._reader, where)
                        for t_idx in range(result.shape[0]):
                            result[t_idx, node] = _graph_value_at(
                                bc._reader, where, t=t, node=node, field=result[t_idx]
                            )
                    else:
                        result[node] = _graph_value_at(bc._reader, where, t=t, node=node, field=result)

        return result

    @staticmethod
    def _value_of(bc: NodeBC, node: int, t: float | None, is_2d: bool) -> float:
        """``bc``'s value at ``node``, where a branch of `apply` writes it: on a 2-D field, not one that reads t (#2296)."""
        if is_2d:
            _refuse_a_time_read_into_rows(bc._reader, f"NodeBC {bc.name!r}: value")
        return bc.get_value(node, t)

    def apply_hjb(
        self,
        u: NDArray[np.floating],
        t: float | None = None,
    ) -> NDArray[np.floating]:
        """
        Apply BCs to HJB value function.

        Convenience wrapper for apply() with field_type="value".

        Args:
            u: Value function at nodes
            t: Current time, needed by a value that reads t (#2296)

        Returns:
            Value function with BCs applied
        """
        return self.apply(u, t, field_type="value")

    def apply_fp(
        self,
        m: NDArray[np.floating],
        t: float | None = None,
    ) -> NDArray[np.floating]:
        """
        Apply BCs to Fokker-Planck density.

        Convenience wrapper for apply() with field_type="density".

        Args:
            m: Density at nodes
            t: Current time, needed by a value that reads t (#2296)

        Returns:
            Density with BCs applied
        """
        return self.apply(m, t, field_type="density")

    # =========================================================================
    # Low-Level API: Direct Node/Edge Specification
    # =========================================================================

    def apply_direct(
        self,
        field: NDArray[np.floating],
        boundary_nodes: list[int],
        values: list[float] | float | Callable[..., float],
        t: float | None = None,
    ) -> NDArray[np.floating]:
        """
        Apply Dirichlet BC directly to specified nodes.

        Low-level API for users who prefer direct control.

        Args:
            field: Node values
            boundary_nodes: List of node indices
            values: BC values: a value (a float, or a callable of (t, node), time first), or one float per node
            t: Current time, needed by a callable that reads t (#2296)

        Returns:
            Field with BCs applied

        Example:
            >>> # Direct specification
            >>> u_bc = applicator.apply_direct(u, [0, 99], [0.0, 1.0])

            >>> # With callable
            >>> u_bc = applicator.apply_direct(u, [0], lambda t, node: np.sin(t), t=0.5)
        """
        result = field.copy()
        nodes = self._checked_nodes(boundary_nodes, "GraphApplicator.apply_direct", "boundary_nodes")
        per_node = _node_values(values, len(nodes), "GraphApplicator.apply_direct")
        read = _value_reader(values, "GraphApplicator.apply_direct: values") if per_node is None else None
        if read is not None and result.ndim == 2:
            _refuse_a_time_read_into_rows(read, "GraphApplicator.apply_direct: values")
        if per_node is not None:
            repeated = sorted(n for n, count in Counter(nodes).items() if count > 1)
            if repeated:
                raise ValueError(
                    f"GraphApplicator.apply_direct: node(s) {repeated} are given more than once, each with a value of "
                    "its own, so only the last would be kept. Give each node once (#2292)."
                )

        for i, node in enumerate(nodes):
            if read is not None:
                value = _graph_value_at(read, "GraphApplicator.apply_direct: values", t=t, node=node)
            else:
                value = per_node[i]

            if result.ndim == 2:
                result[:, node] = value
            else:
                result[node] = value

        return result

    def apply_edge_direct(
        self,
        edge_flow: NDArray[np.floating],
        edges: list[tuple[int, int]],
        values: list[float] | float | Callable[[int, int, float], float],
        t: float = 0.0,
        edge_index_map: dict[tuple[int, int], int] | None = None,
    ) -> NDArray[np.floating]:
        """Refused: no graph solver has an edge field, so an edge value written here reaches no solve (#2493).

        It used to write ``values`` into ``edge_flow``: with no ``edge_index_map``, each edge at its position in ``edges``;
        with one, skipping an edge the map does not list; and repeating the last of a short value list (#2292).
        """
        raise NotImplementedError(f"GraphApplicator.apply_edge_direct: {_EDGES_NOT_APPLIED}")

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_boundary_nodes(self) -> list[int]:
        """
        Get all nodes with Dirichlet or absorbing BC.

        Returns:
            List of boundary node indices
        """
        if self._resolved_boundary_nodes is None:
            self._resolved_boundary_nodes = set()
            for bc in self._node_bcs:
                if bc.bc_type in (GraphBCType.DIRICHLET, GraphBCType.ABSORBING):
                    self._resolved_boundary_nodes.update(self._resolve_nodes(bc))

        return list(self._resolved_boundary_nodes)

    def get_source_nodes(self) -> list[int]:
        """
        Get all nodes with source BC.

        Returns:
            List of source node indices
        """
        if self._resolved_source_nodes is None:
            self._resolved_source_nodes = set()
            for bc in self._node_bcs:
                if bc.bc_type == GraphBCType.SOURCE:
                    self._resolved_source_nodes.update(self._resolve_nodes(bc))

        return list(self._resolved_source_nodes)

    def get_interior_nodes(self) -> list[int]:
        """
        Get nodes that are not on any boundary.

        Returns:
            List of interior node indices
        """
        boundary = set(self.get_boundary_nodes())
        return [i for i in range(self.num_nodes) if i not in boundary]

    def get_bc_by_name(self, name: str) -> NodeBC | None:
        """Get node BC by name, or None if no condition has it. A name several conditions share is refused:
        the ``set_*_nodes`` helpers each give one default name, so two calls of one helper share it (#2292)."""
        named = self._node_bcs_by_name.get(name, [])
        if len(named) > 1:
            raise ValueError(
                f"GraphApplicator.get_bc_by_name: {len(named)} conditions are named {name!r}, so the name picks none "
                "of them. Give each its own name, e.g. set_dirichlet_nodes(..., name=...) (#2292)."
            )
        return named[0] if named else None

    def get_bc_value(self, name: str, node: int, t: float) -> float | None:
        """Get BC value for a specific node from named BC region."""
        node = self._checked_nodes([node], "GraphApplicator.get_bc_value", "node")[0]
        bc = self.get_bc_by_name(name)
        if bc is None:
            return None
        if node in self._resolve_nodes(bc):
            return bc.get_value(node, t)
        return None

    def __repr__(self) -> str:
        """String representation."""
        bc_summary = []
        for bc in self._node_bcs:
            nodes = self._resolve_nodes(bc)
            bc_summary.append(f"{bc.name}({bc.bc_type.value}): {len(nodes)} nodes")

        return f"GraphApplicator(num_nodes={self.num_nodes}, bcs=[{', '.join(bc_summary)}])"


GraphApplicator._WRITES = MappingProxyType(
    {
        bc_type: tuple(
            field for field in ("value", "density") if (bc_type, field) not in GraphApplicator._DECLARED_IDENTITY
        )
        for bc_type in GraphBCType
    }
)

# =============================================================================
# Convenience Functions
# =============================================================================


def _num_nodes(geometry) -> int:
    """A graph geometry's node count. Network geometries call it ``num_nodes``; every ``GraphGeometry``, a maze
    among them, gives ``num_spatial_points`` (#2292)."""
    # Issue #543: getattr() normalizes the attribute naming across geometries
    for name in ("num_nodes", "n_nodes", "num_spatial_points"):
        n = getattr(geometry, name, None)
        if n is not None:
            return int(n)
    raise TypeError(
        f"{type(geometry).__name__} gives no node count: none of num_nodes, n_nodes or num_spatial_points (#2292)."
    )


def accept_graph_boundary_conditions(boundary_conditions, where: str) -> GraphBCConfig | None:
    """The node boundary conditions a graph geometry is handed: a ``GraphBCConfig``, or ``None`` for none.

    Anything else, a continuum ``BoundaryConditions``, a list, or a bare ``NodeBC``, is refused here, by
    name, rather than failing later where the problem is built (#2292).
    """
    if boundary_conditions is None or isinstance(boundary_conditions, GraphBCConfig):
        return boundary_conditions
    raise TypeError(
        f"{where}: boundary_conditions is a {type(boundary_conditions).__name__}. A graph geometry takes a "
        "GraphBCConfig, e.g. GraphBCConfig(node_bcs=[NodeBC(...)]), or None (#2292)."
    )


def create_graph_applicator(
    num_nodes: int,
    boundary_nodes: list[int] | None = None,
    boundary_value: float = 0.0,
    source_nodes: list[int] | None = None,
    source_value: float = 1.0,
    absorbing_nodes: list[int] | None = None,
) -> GraphApplicator:
    """
    Create a graph BC applicator with common settings.

    Convenience function for typical MFG setups.

    Args:
        num_nodes: Total number of nodes
        boundary_nodes: Nodes with Dirichlet BC (default value)
        boundary_value: Value at boundary nodes
        source_nodes: Nodes with source BC (density injection)
        source_value: Source strength
        absorbing_nodes: Nodes where density is absorbed (set to zero)

    Returns:
        Configured GraphApplicator

    Example:
        >>> # Exit at nodes 0, 99; entry at node 50
        >>> applicator = create_graph_applicator(
        ...     num_nodes=100,
        ...     absorbing_nodes=[0, 99],
        ...     source_nodes=[50],
        ...     source_value=1.0
        ... )
    """
    applicator = GraphApplicator(num_nodes)

    # Each read once into a tuple, or refused by name: a truth test on the given object dropped a bare 0 and a
    # one-element array of node 0, and raised on a longer array. None or empty is no condition (#2292).
    _value_reader(boundary_value, "create_graph_applicator: boundary_value")
    _value_reader(source_value, "create_graph_applicator: source_value")
    boundary = _given_nodes(boundary_nodes, "create_graph_applicator: boundary_nodes")
    source = _given_nodes(source_nodes, "create_graph_applicator: source_nodes")
    absorbing = _given_nodes(absorbing_nodes, "create_graph_applicator: absorbing_nodes")
    if boundary:
        applicator.set_dirichlet_nodes(boundary, value=boundary_value, name="boundary")
    if source:
        applicator.set_source_nodes(source, value=source_value, name="source")
    if absorbing:
        applicator.set_absorbing_nodes(absorbing, name="absorbing")

    return applicator


def create_maze_applicator(
    maze_geometry,
    exit_nodes: list[int] | None = None,
    entry_nodes: list[int] | None = None,
    auto_detect_exits: bool = True,
) -> GraphApplicator:
    """
    Create BC applicator for maze problems.

    Args:
        maze_geometry: MazeGeometry or similar maze geometry
        exit_nodes: Explicit exit node indices
        entry_nodes: Explicit entry node indices
        auto_detect_exits: If True, detect exits as leaf nodes (degree 1)

    Returns:
        Configured GraphApplicator for maze

    Example:
        >>> maze = MazeGeometry(rows=10, cols=10)
        >>> applicator = create_maze_applicator(maze, auto_detect_exits=True)
    """
    applicator = GraphApplicator(_num_nodes(maze_geometry))
    # Each read once into a tuple, or refused by name: a generator read twice came back empty, and a truth test
    # dropped a bare 0. None or empty is no condition (#2292).
    given_entries = _given_nodes(entry_nodes, "create_maze_applicator: entry_nodes")
    entry = NodeBC(nodes=given_entries, bc_type=GraphBCType.SOURCE, value=1.0, name="entry") if given_entries else None
    exits: Sequence[int] | Callable[[int], list[int]] | None = _given_nodes(
        exit_nodes, "create_maze_applicator: exit_nodes"
    )

    # Auto-detect exits if none were given. A dead end named as an entry is the entry's: the explicit name overrides
    # the heuristic, where both would have written the density at that node (#2292).
    if auto_detect_exits and exit_nodes is None:
        entries = set() if entry is None else set(applicator._resolve_nodes(entry))
        exits = [n for n in GraphApplicator._detect_leaf_nodes(maze_geometry) if n not in entries]

    if exits:
        applicator.set_absorbing_nodes(exits, name="exit")

    if entry is not None:
        applicator.add_node_bc(entry)

    return applicator


__all__ = [
    # Main class
    "GraphApplicator",
    # Configuration
    "GraphBCConfig",
    "GraphBCType",
    "NodeBC",
    "EdgeBC",
    # Factory functions
    "create_graph_applicator",
    "create_maze_applicator",
]
