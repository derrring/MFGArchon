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

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# Import base class for inheritance
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


@dataclass
class NodeBC:
    """
    Boundary condition specification for a set of nodes.

    Attributes:
        nodes: List of node indices or callable(num_nodes) -> indices
        bc_type: Type of boundary condition
        value: BC value (constant or callable(node, t) -> value)
        name: Optional name for this BC region (e.g., "exit", "source")
    """

    nodes: list[int] | Callable[[int], list[int]]
    bc_type: GraphBCType = GraphBCType.DIRICHLET
    value: float | Callable[[int, float], float] = 0.0
    name: str = "boundary"

    def get_nodes(self, num_nodes: int) -> list[int]:
        """Get node indices, resolving callable if needed."""
        if callable(self.nodes):
            return self.nodes(num_nodes)
        return list(self.nodes)

    def get_value(self, node: int, t: float) -> float:
        """Get BC value for a specific node at time t."""
        if callable(self.value):
            return self.value(node, t)
        return float(self.value)


@dataclass
class EdgeBC:
    """
    Boundary condition specification for edges (flow-based problems).

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

    def get_edges(self, num_nodes: int) -> list[tuple[int, int]]:
        """Get edge tuples, resolving callable if needed."""
        if callable(self.edges):
            return self.edges(num_nodes)
        return list(self.edges)

    def get_value(self, from_node: int, to_node: int, t: float) -> float:
        """Get BC value for a specific edge at time t."""
        if callable(self.value):
            return self.value(from_node, to_node, t)
        return float(self.value)


@dataclass
class GraphBCConfig:
    """
    Configuration for graph boundary conditions.

    Provides a declarative way to specify BCs for graph problems.

    Example:
        >>> config = GraphBCConfig(
        ...     node_bcs=[
        ...         NodeBC(nodes=[0, 1], bc_type=GraphBCType.DIRICHLET, value=0.0, name="exit"),
        ...         NodeBC(nodes=[10, 11], bc_type=GraphBCType.SOURCE, value=1.0, name="source"),
        ...     ],
        ...     default_bc_type=GraphBCType.NEUMANN
        ... )
        >>> applicator = GraphApplicator.from_config(config, num_nodes=20)
    """

    node_bcs: list[NodeBC] = field(default_factory=list)
    edge_bcs: list[EdgeBC] = field(default_factory=list)
    default_bc_type: GraphBCType = GraphBCType.NEUMANN


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

        >>> # Direct edge flow specification
        >>> flow_bc = applicator.apply_edge_direct(flow, edges=[(0,1), (2,3)], values=[0.0, 0.0])
    """

    def __init__(self, num_nodes: int):
        """
        Initialize graph BC applicator.

        Args:
            num_nodes: Total number of nodes in the graph
        """
        super().__init__(num_nodes)

        # Node BC storage
        self._node_bcs: list[NodeBC] = []
        self._node_bc_map: dict[str, NodeBC] = {}

        # Edge BC storage
        self._edge_bcs: list[EdgeBC] = []
        self._edge_bc_map: dict[str, EdgeBC] = {}

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
        applicator = cls(num_nodes)
        for node_bc in config.node_bcs:
            applicator.add_node_bc(node_bc)
        for edge_bc in config.edge_bcs:
            applicator.add_edge_bc(edge_bc)
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
                - boundary_nodes: For "custom" type

        Returns:
            Configured GraphApplicator

        Example:
            >>> network = GridNetwork(10, 10)
            >>> applicator = GraphApplicator.from_network_geometry(network, boundary_type="leaf")
        """
        num_nodes = geometry.num_nodes

        applicator = cls(num_nodes)

        # Detect boundary nodes based on type
        if boundary_type == "leaf":
            boundary_nodes = cls._detect_leaf_nodes(geometry)
        elif boundary_type == "degree":
            threshold = kwargs.get("degree_threshold", 1)
            boundary_nodes = cls._detect_low_degree_nodes(geometry, threshold)
        elif boundary_type == "spatial":
            boundary_nodes = cls._detect_spatial_boundary_nodes(geometry)
        elif boundary_type == "custom":
            boundary_nodes = kwargs.get("boundary_nodes", [])
        else:
            raise ValueError(f"Unknown boundary_type: {boundary_type}")

        if boundary_nodes:
            applicator.set_dirichlet_nodes(boundary_nodes, value=kwargs.get("boundary_value", 0.0), name="boundary")

        return applicator

    @staticmethod
    def _detect_leaf_nodes(geometry) -> list[int]:
        """Detect leaf nodes (degree 1) in the graph."""
        # Issue #543: Use try/except instead of hasattr() for optional attribute
        try:
            network_data = geometry.network_data
            if network_data is not None:
                adj = network_data.adjacency_matrix
                if adj is not None:
                    degrees = np.array(adj.sum(axis=1)).flatten()
                    return list(np.where(degrees == 1)[0])
        except AttributeError:
            pass
        return []

    @staticmethod
    def _detect_low_degree_nodes(geometry, threshold: int) -> list[int]:
        """Detect nodes with degree <= threshold."""
        # Issue #543: Use try/except instead of hasattr() for optional attribute
        try:
            network_data = geometry.network_data
            if network_data is not None:
                adj = network_data.adjacency_matrix
                if adj is not None:
                    degrees = np.array(adj.sum(axis=1)).flatten()
                    return list(np.where(degrees <= threshold)[0])
        except AttributeError:
            pass
        return []

    @staticmethod
    def _detect_spatial_boundary_nodes(geometry, tolerance: float = 1e-6) -> list[int]:
        """Detect nodes on spatial bounding box (for spatially-embedded networks)."""
        # Issue #543: Use try/except instead of hasattr() for optional attribute
        try:
            network_data = geometry.network_data
            if network_data is not None:
                positions = network_data.node_positions
                if positions is not None:
                    min_coords = np.min(positions, axis=0)
                    max_coords = np.max(positions, axis=0)

                    boundary_nodes = []
                    for i, pos in enumerate(positions):
                        # Check if on any boundary face
                        on_boundary = np.any(np.abs(pos - min_coords) < tolerance) or np.any(
                            np.abs(pos - max_coords) < tolerance
                        )
                        if on_boundary:
                            boundary_nodes.append(i)
                    return boundary_nodes
        except AttributeError:
            pass
        return []

    # =========================================================================
    # High-Level API: Named Region Specification
    # =========================================================================

    def set_dirichlet_nodes(
        self,
        nodes: list[int] | Callable[[int], list[int]],
        value: float | Callable[[int, float], float] = 0.0,
        name: str = "dirichlet",
    ) -> GraphApplicator:
        """
        Set Dirichlet BC on specified nodes.

        Args:
            nodes: Node indices or callable(num_nodes) -> indices
            value: BC value (constant or callable(node, t) -> value)
            name: Name for this BC region

        Returns:
            self (for method chaining)

        Example:
            >>> applicator.set_dirichlet_nodes([0, 99], value=0.0, name="exit")
            >>> applicator.set_dirichlet_nodes(
            ...     nodes=[10],
            ...     value=lambda node, t: np.sin(t),
            ...     name="oscillating"
            ... )
        """
        bc = NodeBC(nodes=nodes, bc_type=GraphBCType.DIRICHLET, value=value, name=name)
        return self.add_node_bc(bc)

    def set_absorbing_nodes(
        self,
        nodes: list[int] | Callable[[int], list[int]],
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
        nodes: list[int] | Callable[[int], list[int]],
        value: float | Callable[[int, float], float] = 1.0,
        name: str = "source",
    ) -> GraphApplicator:
        """
        Set source BC on specified nodes (density injection).

        Args:
            nodes: Node indices or callable
            value: Source strength (constant or callable)
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
        nodes: list[int] | Callable[[int], list[int]],
        bc_func: Callable[[NDArray, int, float], float],
        name: str = "custom",
    ) -> GraphApplicator:
        """
        Set custom BC with user-defined function.

        Args:
            nodes: Node indices or callable
            bc_func: Function(field, node, t) -> new_value
            name: Name for this BC region

        Returns:
            self (for method chaining)

        Example:
            >>> def robin_bc(field, node, t):
            ...     # Robin-like: u + alpha * du/dn = g
            ...     return 0.5 * field[node]
            >>> applicator.set_custom_bc([0, 99], robin_bc)
        """

        # Wrap bc_func to match NodeBC value signature
        def value_func(node: int, t: float) -> float:
            # Note: This requires field to be passed separately in apply()
            return 0.0  # Placeholder - actual value set during apply

        bc = NodeBC(nodes=nodes, bc_type=GraphBCType.CUSTOM, value=bc_func, name=name)
        return self.add_node_bc(bc)

    def add_node_bc(self, bc: NodeBC) -> GraphApplicator:
        """Add a node BC specification."""
        self._node_bcs.append(bc)
        self._node_bc_map[bc.name] = bc
        self._invalidate_cache()
        return self

    def add_edge_bc(self, bc: EdgeBC) -> GraphApplicator:
        """Add an edge BC specification."""
        self._edge_bcs.append(bc)
        self._edge_bc_map[bc.name] = bc
        return self

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
        t: float = 0.0,
        field_type: Literal["value", "density"] = "value",
    ) -> NDArray[np.floating]:
        """
        Apply all configured boundary conditions to a field.

        Args:
            field: Node values (shape: (num_nodes,) or (Nt, num_nodes))
            t: Current time
            field_type: Type of field
                - "value": HJB value function (u)
                - "density": FP density (m)

        Returns:
            Field with BCs applied

        Example:
            >>> u = np.ones(100)
            >>> u_bc = applicator.apply(u, t=0.5, field_type="value")
        """
        result = field.copy()
        is_2d = result.ndim == 2

        for bc in self._node_bcs:
            nodes = bc.get_nodes(self.num_nodes)

            for node in nodes:
                if node < 0 or node >= self.num_nodes:
                    continue

                if bc.bc_type == GraphBCType.DIRICHLET:
                    # Issue #1471: a Dirichlet value pin belongs to the VALUE field (HJB u), not the
                    # density field (FP m) — pinning m to a value-function value is wrong.
                    if field_type == "value":
                        value = bc.get_value(node, t)
                        if is_2d:
                            result[:, node] = value
                        else:
                            result[node] = value

                elif bc.bc_type == GraphBCType.ABSORBING:
                    # Issue #1478: an absorbing / exit node is ONE physical BC with dual operations
                    # (adjoint duality): the HJB value carries the exit cost (Dirichlet u = value), and
                    # the FP density is absorbed (m -> 0, mass exits). Apply the right one per field.
                    if field_type == "value":
                        value = bc.get_value(node, t)
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
                        value = bc.get_value(node, t)
                        if is_2d:
                            result[:, node] = value
                        else:
                            result[node] = value
                    # For value functions, source nodes don't modify u

                elif bc.bc_type == GraphBCType.NEUMANN:
                    # No change for Neumann (handled by solver)
                    pass

                elif bc.bc_type == GraphBCType.CUSTOM:
                    if callable(bc.value):
                        if is_2d:
                            for t_idx in range(result.shape[0]):
                                result[t_idx, node] = bc.value(result[t_idx], node, t)
                        else:
                            result[node] = bc.value(result, node, t)

        return result

    def apply_hjb(
        self,
        u: NDArray[np.floating],
        t: float = 0.0,
    ) -> NDArray[np.floating]:
        """
        Apply BCs to HJB value function.

        Convenience wrapper for apply() with field_type="value".

        Args:
            u: Value function at nodes
            t: Current time

        Returns:
            Value function with BCs applied
        """
        return self.apply(u, t, field_type="value")

    def apply_fp(
        self,
        m: NDArray[np.floating],
        t: float = 0.0,
    ) -> NDArray[np.floating]:
        """
        Apply BCs to Fokker-Planck density.

        Convenience wrapper for apply() with field_type="density".

        Args:
            m: Density at nodes
            t: Current time

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
        values: list[float] | float | Callable[[int, float], float],
        t: float = 0.0,
    ) -> NDArray[np.floating]:
        """
        Apply Dirichlet BC directly to specified nodes.

        Low-level API for users who prefer direct control.

        Args:
            field: Node values
            boundary_nodes: List of node indices
            values: BC values (single float, list, or callable(node, t))
            t: Current time

        Returns:
            Field with BCs applied

        Example:
            >>> # Direct specification
            >>> u_bc = applicator.apply_direct(u, [0, 99], [0.0, 1.0])

            >>> # With callable
            >>> u_bc = applicator.apply_direct(u, [0], lambda n, t: np.sin(t))
        """
        result = field.copy()

        for i, node in enumerate(boundary_nodes):
            if node < 0 or node >= self.num_nodes:
                continue

            if callable(values):
                value = values(node, t)
            elif isinstance(values, (list, np.ndarray)):
                value = values[i] if i < len(values) else values[-1]
            else:
                value = float(values)

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
        """
        Apply BC directly to edge flows.

        Low-level API for flow-based problems.

        Args:
            edge_flow: Edge flow values (shape: (num_edges,))
            edges: List of (from_node, to_node) tuples
            values: BC values for edges
            t: Current time
            edge_index_map: Mapping from (from, to) to edge index

        Returns:
            Edge flow with BCs applied

        Example:
            >>> flow_bc = applicator.apply_edge_direct(
            ...     flow, edges=[(0,1), (99,98)], values=0.0
            ... )
        """
        result = edge_flow.copy()

        if edge_index_map is None:
            # Assume edges are indexed in order
            edge_index_map = {edge: i for i, edge in enumerate(edges)}

        for i, edge in enumerate(edges):
            edge_idx = edge_index_map.get(edge)
            if edge_idx is None:
                continue

            if callable(values):
                value = values(edge[0], edge[1], t)
            elif isinstance(values, (list, np.ndarray)):
                value = values[i] if i < len(values) else values[-1]
            else:
                value = float(values)

            result[edge_idx] = value

        return result

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
                    self._resolved_boundary_nodes.update(bc.get_nodes(self.num_nodes))

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
                    self._resolved_source_nodes.update(bc.get_nodes(self.num_nodes))

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
        """Get node BC by name."""
        return self._node_bc_map.get(name)

    def get_bc_value(self, name: str, node: int, t: float) -> float | None:
        """Get BC value for a specific node from named BC region."""
        bc = self.get_bc_by_name(name)
        if bc is None:
            return None
        nodes = bc.get_nodes(self.num_nodes)
        if node in nodes:
            return bc.get_value(node, t)
        return None

    def __repr__(self) -> str:
        """String representation."""
        bc_summary = []
        for bc in self._node_bcs:
            nodes = bc.get_nodes(self.num_nodes)
            bc_summary.append(f"{bc.name}({bc.bc_type.value}): {len(nodes)} nodes")

        return f"GraphApplicator(num_nodes={self.num_nodes}, bcs=[{', '.join(bc_summary)}])"


# =============================================================================
# Convenience Functions
# =============================================================================


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

    if boundary_nodes:
        applicator.set_dirichlet_nodes(boundary_nodes, value=boundary_value, name="boundary")

    if source_nodes:
        applicator.set_source_nodes(source_nodes, value=source_value, name="source")

    if absorbing_nodes:
        applicator.set_absorbing_nodes(absorbing_nodes, name="absorbing")

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
        >>> maze = MazeGeometry(width=10, height=10)
        >>> applicator = create_maze_applicator(maze, auto_detect_exits=True)
    """
    # Get number of nodes from geometry
    # Issue #543: Use getattr() to normalize attribute naming (num_nodes vs n_nodes)
    num_nodes = getattr(maze_geometry, "num_nodes", None) or getattr(maze_geometry, "n_nodes", None)
    if num_nodes is None:
        raise ValueError(
            "Cannot determine number of nodes from maze geometry. "
            "Geometry must have 'num_nodes' or 'n_nodes' attribute."
        )

    applicator = GraphApplicator(num_nodes)

    # Auto-detect exits if requested
    if auto_detect_exits and exit_nodes is None:
        exit_nodes = GraphApplicator._detect_leaf_nodes(maze_geometry)

    if exit_nodes:
        applicator.set_absorbing_nodes(exit_nodes, name="exit")

    if entry_nodes:
        applicator.set_source_nodes(entry_nodes, value=1.0, name="entry")

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
