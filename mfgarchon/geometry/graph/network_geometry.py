"""
Network Geometry for Mean Field Games on Graphs.

This module implements geometric representations for MFG problems on network/graph structures,
extending the MFG framework from continuous domains to discrete network topologies.

Key concepts:
- Nodes: Discrete locations where players can position themselves
- Edges: Connections between nodes allowing player movement
- Network flows: Discrete analogue of continuous density evolution
- Graph Laplacians: Discrete operators for diffusion and gradient terms
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import csr_matrix, diags

from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# Import base geometry class
from mfgarchon.geometry.base import GraphGeometry
from mfgarchon.geometry.boundary.tolerances import ONWALL_TOL

# Import geometry protocol
from mfgarchon.geometry.protocol import GeometryType

# Import unified backend system
from .network_backend import (
    NetworkBackendType,
    OperationType,
    get_backend_manager,
)

# Legacy compatibility - keep old imports for fallback
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    import igraph as ig

    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    ig = None

# Module logger
logger = get_logger(__name__)


class NetworkType(Enum):
    """Types of network structures for MFG problems."""

    GRID = "grid"
    RANDOM = "random"
    SCALE_FREE = "scale_free"
    SMALL_WORLD = "small_world"
    COMPLETE = "complete"
    TREE = "tree"
    LATTICE = "lattice"
    CUSTOM = "custom"


@dataclass
class NetworkData:
    """
    Network data container for MFG on graphs.

    This class stores all geometric and topological information needed
    to solve MFG problems on network structures.

    Attributes:
        adjacency_matrix: (N, N) sparse adjacency matrix
        node_positions: (N, d) coordinates for visualization [optional]
        edge_weights: Edge weight vector [optional, defaults to 1]
        node_weights: Node weight vector [optional, defaults to 1]
        laplacian_matrix: Graph Laplacian for diffusion operators
        degree_matrix: Node degree matrix
        incidence_matrix: Node-edge incidence matrix
        num_nodes: Number of nodes in network
        num_edges: Number of edges in network
        network_type: Type of network structure
        metadata: Additional network properties
    """

    adjacency_matrix: csr_matrix  # Required field, never None
    num_nodes: int
    num_edges: int
    network_type: NetworkType = NetworkType.CUSTOM

    # Optional geometric information
    node_positions: np.ndarray | None = None  # (N, d) coordinates
    edge_weights: np.ndarray | None = None  # Edge weights
    node_weights: np.ndarray | None = None  # Node weights/capacities

    # Derived matrices (computed automatically)
    laplacian_matrix: csr_matrix | None = None
    degree_matrix: csr_matrix | None = None
    incidence_matrix: csr_matrix | None = None

    # Network properties
    is_directed: bool = False
    is_weighted: bool = False
    is_connected: bool = True

    # Additional data
    metadata: dict[str, Any] = field(default_factory=dict)

    # Backend information
    backend_type: NetworkBackendType | None = None
    backend_graph: Any | None = None  # Original backend graph object

    def __post_init__(self):
        """Compute derived network properties and matrices."""
        self._validate_network_data()
        self._compute_derived_matrices()
        self._compute_network_properties()

    def _validate_network_data(self):
        """Validate network data consistency."""
        # Validate adjacency matrix
        adjacency = self.adjacency_matrix
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        if adjacency.shape[0] != self.num_nodes:
            raise ValueError(f"Adjacency matrix size {adjacency.shape[0]} != num_nodes {self.num_nodes}")

        # Validate node positions with explicit type narrowing
        node_positions = self.node_positions
        if node_positions is not None:
            if node_positions.shape[0] != self.num_nodes:
                raise ValueError("Node positions must match number of nodes")

        # Validate edge weights with explicit type narrowing
        edge_weights = self.edge_weights
        if edge_weights is not None:
            if len(edge_weights) != self.num_edges:
                raise ValueError("Edge weights must match number of edges")

        # Validate node weights with explicit type narrowing
        node_weights = self.node_weights
        if node_weights is not None:
            if len(node_weights) != self.num_nodes:
                raise ValueError("Node weights must match number of nodes")

    def _compute_derived_matrices(self):
        """Compute graph Laplacian and other derived matrices."""
        A = self.adjacency_matrix

        # Default edge weights
        if self.edge_weights is None:
            self.edge_weights = np.ones(self.num_edges)

        # Default node weights
        if self.node_weights is None:
            self.node_weights = np.ones(self.num_nodes)

        # Degree matrix
        degrees = np.array(A.sum(axis=1)).flatten()
        self.degree_matrix = csr_matrix(diags(degrees, format="csr"))

        # Graph Laplacian (L = D - A)
        self.laplacian_matrix = self.degree_matrix - A

        # Incidence matrix (for edge-based operations)
        self.incidence_matrix = self._compute_incidence_matrix()

    def _compute_incidence_matrix(self) -> csr_matrix:
        """Compute node-edge incidence matrix."""
        rows, cols = [], []
        data = []

        edge_idx = 0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.adjacency_matrix[i, j] != 0:
                    # Edge connects nodes i and j
                    rows.extend([i, j])
                    cols.extend([edge_idx, edge_idx])
                    data.extend([1, -1])  # Oriented incidence
                    edge_idx += 1

        return csr_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_edges))

    def _compute_network_properties(self):
        """Compute basic network properties."""
        A = self.adjacency_matrix

        # Check if directed
        self.is_directed = (A != A.T).nnz != 0

        # Check if weighted
        self.is_weighted = not np.allclose(A.data, 1.0)

        # Check connectivity (simplified check)
        self.is_connected = self._check_connectivity()

        # Store additional properties with safe metadata access
        metadata = getattr(self, "metadata", {})
        metadata.update(
            {
                "average_degree": np.mean(np.array(A.sum(axis=1)).flatten()),
                "max_degree": np.max(np.array(A.sum(axis=1)).flatten()),
                "density": self.num_edges / (self.num_nodes * (self.num_nodes - 1) / 2),
                "clustering_coefficient": self._compute_clustering_coefficient(),
            }
        )

    def _check_connectivity(self) -> bool:
        """Check if network is connected (simplified version)."""
        if not NETWORKX_AVAILABLE or nx is None:
            return True  # Assume connected if NetworkX not available

        # Convert to NetworkX for connectivity check
        G = nx.from_scipy_sparse_array(self.adjacency_matrix)
        return nx.is_connected(G) if not self.is_directed else nx.is_strongly_connected(G)

    def _compute_clustering_coefficient(self) -> float:
        """Compute average clustering coefficient."""
        if not NETWORKX_AVAILABLE or nx is None:
            return 0.0

        G = nx.from_scipy_sparse_array(self.adjacency_matrix)
        return nx.average_clustering(G)

    def get_neighbors(self, node_id: int) -> list[int]:
        """Get neighbors of a given node."""
        return self.adjacency_matrix.getrow(node_id).nonzero()[1].tolist()

    def get_edge_weight(self, node_i: int, node_j: int) -> float:
        """Get weight of edge between two nodes."""
        return self.adjacency_matrix[node_i, node_j]

    def get_node_degree(self, node_id: int) -> int:
        """Get degree of a node."""
        return int(self.adjacency_matrix.getrow(node_id).sum())


class NetworkGeometry(GraphGeometry):
    """
    Abstract base class for network geometries in MFG problems.

    This class defines the interface for creating and managing
    network structures for Mean Field Games on graphs.

    Inherits from GraphGeometry to participate in the geometry hierarchy:
        Geometry -> GraphGeometry -> NetworkGeometry -> GridNetwork, etc.

    Implemented Geometry Traits (Issue #596 Phase 2.4):
        - SupportsGraphLaplacian: Discrete Laplacian operator L = D - A for diffusion
        - SupportsAdjacency: Adjacency matrix and neighbor queries
        - SupportsSpatialEmbedding: Node positions for positioned graphs (optional)
        - SupportsGraphDistance: Shortest path distances on graph topology

    Compatible Solvers:
        - Network HJB solvers (using graph Laplacian for diffusion)
        - Network FP solvers (using graph operators for density evolution)
        - Network MFG solvers (coupling HJB and FP on graphs)

    Key features:
        - Unified interface with MazeGeometry and other graph types
        - Optional spatial embedding (node positions)
        - Multiple backend support (igraph, networkx, networkit)
        - Sparse matrix operations for efficiency
        - Trait-based operator access for solver integration
    """

    def __init__(
        self,
        num_nodes: int,
        network_type: NetworkType = NetworkType.CUSTOM,
        backend_preference: NetworkBackendType = NetworkBackendType.IGRAPH,
        boundary_conditions=None,
    ):
        super().__init__(boundary_conditions=boundary_conditions)  # Issue #1471: node-BC on the geometry
        self._num_nodes = num_nodes
        self.network_type = network_type
        self.network_data: NetworkData | None = None
        self.backend_preference = backend_preference
        self.backend_manager = get_backend_manager(backend_preference)
        self._regions: dict[str, np.ndarray] = {}

    # =========================================================================
    # Pickle Support (Phase 5 of Issue #435)
    # =========================================================================

    def __getstate__(self) -> dict[str, Any]:
        """
        Get state for pickling.

        Excludes backend_manager which contains module references that
        cannot be pickled. It will be recreated on unpickling.
        """
        state = self.__dict__.copy()
        # Remove unpickleable backend manager
        state.pop("backend_manager", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore state from pickle.

        Recreates backend_manager from backend_preference.
        """
        self.__dict__.update(state)
        # Recreate backend manager
        backend_pref = getattr(self, "backend_preference", NetworkBackendType.IGRAPH)
        self.backend_manager = get_backend_manager(backend_pref)

    # =========================================================================
    # GraphGeometry abstract method implementations
    # =========================================================================

    @property
    def dimension(self) -> int:
        """
        Spatial dimension of network geometry.

        For networks, dimension represents the embedding space dimension
        if node_positions are available, or 1 (topological dimension) otherwise.

        Note: This overrides GraphGeometry.dimension (which returns 0 for
        topological dimension) because networks often have spatial embedding.
        """
        if self.network_data is not None and self.network_data.node_positions is not None:
            return self.network_data.node_positions.shape[1]
        return 1  # Default for abstract networks

    @property
    def geometry_type(self) -> GeometryType:
        """Type of geometry (always NETWORK for network geometries)."""
        return GeometryType.NETWORK

    @property
    def num_spatial_points(self) -> int:
        """Total number of discrete spatial points (network nodes)."""
        return self._num_nodes

    # Alias for backward compatibility
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the network (alias for num_spatial_points)."""
        return self._num_nodes

    @abstractmethod
    def get_adjacency_matrix(self) -> NDArray:
        """
        Get adjacency matrix for the network (SupportsAdjacency trait).

        Returns:
            Adjacency matrix A of shape (N, N) where:
                A[i,j] = weight of edge from node i to node j
                A[i,j] = 0 if no edge exists

        Note: Required by GraphGeometry ABC and SupportsAdjacency protocol.
        Concrete implementations should return the adjacency matrix from their network_data.
        """
        ...

    def get_neighbors(self, node_idx: int) -> list[int]:
        """
        Get neighbor indices for a node (SupportsAdjacency trait).

        Args:
            node_idx: Node index (0 ≤ node_idx < N)

        Returns:
            List of neighbor node indices (nodes connected by edges)

        Raises:
            IndexError: If node_idx out of range
            ValueError: If network data not initialized

        Example:
            >>> network = GridNetwork(width=5, height=5)
            >>> network.create_network()
            >>> neighbors = network.get_neighbors(node_idx=5)
            >>> # [0, 4, 6, 10] - nodes connected to node 5 in grid
        """
        if self.network_data is None:
            raise ValueError("Network data not initialized. Call create_network() first.")

        return self.network_data.get_neighbors(node_idx)

    def get_spatial_grid(self) -> np.ndarray:
        """
        Get spatial grid representation.

        Returns:
            For networks with positions: (num_nodes, dimension) position array
            For abstract networks: (num_nodes, num_nodes) adjacency matrix
        """
        if self.network_data is None:
            raise ValueError("Network data not initialized. Call create_network() first.")

        # Return node positions if available
        if self.network_data.node_positions is not None:
            return self.network_data.node_positions

        # Otherwise return adjacency matrix as network structure representation
        return self.network_data.adjacency_matrix.toarray()

    @abstractmethod
    def create_network(self, **kwargs) -> NetworkData:
        """Create network with specified parameters."""

    @abstractmethod
    def compute_distance_matrix(self) -> np.ndarray:
        """
        Compute shortest path distances between all node pairs.

        Returns:
            Distance matrix D of shape (N, N) where D[i, j] = shortest path from i to j

        Note: This abstract method is implemented by subclasses using their preferred
        backend (NetworkX, iGraph, etc.). For protocol-compliant interface,
        use compute_all_pairs_distance() which wraps this method.
        """

    def get_graph_distance(
        self,
        node_i: int,
        node_j: int,
        weighted: bool = False,
    ) -> float:
        """
        Compute graph distance (shortest path length) between nodes (SupportsGraphDistance trait).

        Args:
            node_i: Source node index
            node_j: Target node index
            weighted: If True, use edge weights for path cost
                     If False, count hops (all edges have weight 1)

        Returns:
            Shortest path length from node_i to node_j
                - Returns inf if no path exists

        Example:
            >>> network = GridNetwork(width=5, height=5)
            >>> network.create_network()
            >>> # Manhattan distance on grid
            >>> hops = network.get_graph_distance(0, 24, weighted=False)
            >>> # 8 hops (from top-left to bottom-right on 5x5 grid)

        Note:
            For better performance when computing many distances, use
            compute_all_pairs_distance() to get the full distance matrix.
        """
        # Use precomputed distance matrix if available
        distance_matrix = self.compute_distance_matrix()
        return float(distance_matrix[node_i, node_j])

    def compute_all_pairs_distance(
        self,
        weighted: bool = False,
    ) -> np.ndarray:
        """
        Compute distance matrix for all node pairs (SupportsGraphDistance trait).

        Args:
            weighted: If True, use edge weights. If False, count hops.

        Returns:
            Distance matrix D of shape (N, N) where D[i, j] = distance from i to j

        Example:
            >>> network = GridNetwork(width=5, height=5)
            >>> network.create_network()
            >>> D = network.compute_all_pairs_distance(weighted=False)
            >>> diameter = np.max(D[D < np.inf])  # Graph diameter
            >>> # 8 (maximum distance in 5x5 grid)

        Note:
            This method wraps compute_distance_matrix() with protocol-compliant signature.
            The weighted parameter is provided for protocol compatibility but may not
            affect all implementations (depends on backend).
        """
        return self.compute_distance_matrix()

    def get_node_positions(self) -> NDArray | None:
        """
        Get physical coordinates for spatially-embedded networks (SupportsSpatialEmbedding trait).

        Returns:
            (N, d) array of node positions in R^d, or None if abstract network

        Note: Overrides GraphGeometry.get_node_positions() which returns None.
        Implements SupportsSpatialEmbedding protocol for positioned graphs.
        """
        if self.network_data is not None:
            return self.network_data.node_positions
        return None

    def get_euclidean_distance(
        self,
        node_i: int,
        node_j: int,
    ) -> float:
        """
        Compute Euclidean distance between nodes in embedding space (SupportsSpatialEmbedding trait).

        Args:
            node_i: First node index
            node_j: Second node index

        Returns:
            Euclidean distance |xᵢ - xⱼ| in embedding space

        Raises:
            ValueError: If network has no spatial embedding
            IndexError: If node indices out of range

        Note:
            This is DIFFERENT from graph distance (shortest path length).
            Euclidean distance is "as the crow flies", while graph distance
            respects network topology and can only travel along edges.

        Example:
            >>> network = GridNetwork(width=5, height=5)
            >>> network.create_network()
            >>> # Nodes at (0,0) and (3,4) in grid
            >>> dist = network.get_euclidean_distance(0, 19)
            >>> # 5.0 (Euclidean distance √(3² + 4²))
        """
        positions = self.get_node_positions()
        if positions is None:
            raise ValueError("Network has no spatial embedding. Cannot compute Euclidean distance.")

        if not (0 <= node_i < len(positions) and 0 <= node_j < len(positions)):
            raise IndexError(f"Node indices out of range: node_i={node_i}, node_j={node_j}")

        return float(np.linalg.norm(positions[node_i] - positions[node_j]))

    def get_laplacian_operator(self) -> Callable[[NDArray, int], float]:
        """
        Return graph Laplacian operator for discrete diffusion.

        Returns:
            Function with signature: (u: NDArray, node_idx: int) -> float
            Computing: (L u)[node_idx] = sum_j L[node_idx, j] * u[j]

        Note: This is compatible with GraphGeometry's interface.
        For sparse matrix operations, use get_sparse_laplacian() instead.
        """
        L = self.get_sparse_laplacian()  # Get sparse matrix form

        def graph_laplacian_op(u: NDArray, node_idx: int) -> float:
            """
            Apply graph Laplacian to node.

            Args:
                u: Solution vector at nodes (N,)
                node_idx: Node index

            Returns:
                Laplacian value: (L u)[node_idx]
            """
            return float(L.getrow(node_idx).dot(u))

        return graph_laplacian_op

    def get_graph_laplacian_operator(
        self,
        normalized: bool = False,
    ) -> csr_matrix:
        """
        Return discrete graph Laplacian operator (SupportsGraphLaplacian trait).

        This method implements the SupportsGraphLaplacian protocol for trait-based
        solver design. It provides the graph Laplacian in sparse matrix form for
        efficient discrete diffusion operations.

        Args:
            normalized: If True, return normalized Laplacian L_norm = I - D^(-1/2) A D^(-1/2)
                       If False, return unnormalized L = D - A (default)

        Returns:
            Sparse Laplacian matrix L of shape (N, N) where N is number of nodes

        Mathematical Definition:
            - Unnormalized: L = D - A
            - Normalized: L_norm = I - D^(-1/2) A D^(-1/2)

        where:
            - D = diag(∑ⱼ Aᵢⱼ) is the degree matrix
            - A is the adjacency matrix

        Used For:
            - Discrete diffusion: ∂m/∂t = -σ²Lm
            - Graph-based PDEs
            - Spectral graph analysis

        Example:
            >>> network = GridNetwork(width=10, height=10)
            >>> network.create_network()
            >>> L = network.get_graph_laplacian_operator(normalized=False)
            >>> # Discrete heat equation on graph
            >>> m_new = m - dt * sigma**2 * (L @ m)

        Note:
            This method wraps get_sparse_laplacian() with protocol-compliant signature.
            For more Laplacian types (random_walk, etc.), use get_sparse_laplacian().

        Raises:
            ValueError: If network data not initialized
        """
        operator_type = "normalized" if normalized else "combinatorial"
        return self.get_sparse_laplacian(operator_type=operator_type)

    def get_sparse_laplacian(self, operator_type: str = "combinatorial") -> csr_matrix:
        """
        Get graph Laplacian as sparse matrix for efficient computations.

        This method provides direct access to the sparse Laplacian matrix,
        which is more efficient for matrix operations than the callable
        interface from get_laplacian_operator().

        Args:
            operator_type: Type of Laplacian:
                - "combinatorial": L = D - A (default)
                - "normalized": L_norm = D^(-1/2) L D^(-1/2)
                - "random_walk": L_rw = D^(-1) L

        Returns:
            Sparse Laplacian matrix (csr_matrix)

        Raises:
            ValueError: If network data not initialized or invalid operator_type
        """
        if self.network_data is None:
            raise ValueError("Network data not initialized. Call create_network() first.")

        # Type-safe attribute access after null check
        network_data = self.network_data
        A = network_data.adjacency_matrix
        D = network_data.degree_matrix

        if D is None:
            raise ValueError("Degree matrix is required for Laplacian computation")

        if operator_type == "combinatorial":
            return D - A
        elif operator_type == "normalized":
            # L_norm = D^(-1/2) * L * D^(-1/2)
            D_inv_sqrt = csr_matrix(diags(1.0 / np.sqrt(D.diagonal() + 1e-12), format="csr"))
            L = D - A
            return D_inv_sqrt @ L @ D_inv_sqrt
        elif operator_type == "random_walk":
            # L_rw = D^(-1) * L
            D_inv = csr_matrix(diags(1.0 / (D.diagonal() + 1e-12), format="csr"))
            L = D - A
            return D_inv @ L
        else:
            raise ValueError(f"Unknown Laplacian type: {operator_type}")

    def _create_network_data_from_backend(
        self,
        backend_graph: Any,
        backend_type: NetworkBackendType,
        node_positions: np.ndarray | None = None,
    ) -> NetworkData:
        """
        Convert backend graph to NetworkData format.

        Args:
            backend_graph: Graph object from backend
            backend_type: Which backend was used
            node_positions: Optional node positions for visualization

        Returns:
            NetworkData object with all matrices computed
        """
        # Type-safe backend access
        backend_manager = self.backend_manager
        backend = backend_manager.get_backend(backend_type)

        # Get adjacency matrix
        adjacency_matrix = backend.get_adjacency_matrix(backend_graph)

        # Issue #543: Use try/except instead of hasattr() for optional backend method
        # Count edges (handle directed/undirected)
        try:
            is_directed = backend_graph.is_directed()
        except AttributeError:
            # Check if matrix is symmetric
            is_directed = (adjacency_matrix != adjacency_matrix.T).nnz != 0

        num_edges = adjacency_matrix.nnz if is_directed else adjacency_matrix.nnz // 2

        # Create NetworkData
        network_data = NetworkData(
            adjacency_matrix=adjacency_matrix,
            num_nodes=self.num_nodes,
            num_edges=num_edges,
            network_type=self.network_type,
            node_positions=node_positions,
            is_directed=is_directed,
            backend_type=backend_type,
            backend_graph=backend_graph,
        )

        return network_data

    def visualize_network(
        self,
        node_values: np.ndarray | None = None,
        edge_values: np.ndarray | None = None,
        **kwargs,
    ) -> Any:
        """Visualize network structure with optional node/edge values."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for network visualization")

        # Implementation will be in visualization module

    # =========================================================================
    # Boundary Methods (GeometryProtocol - mandatory)
    # =========================================================================

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Get bounding box for spatially-embedded networks.

        Returns:
            (min_coords, max_coords) tuple, or None if abstract network
        """
        if self.network_data is None:
            return None

        positions = self.network_data.node_positions
        if positions is None:
            return None

        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        return min_coords, max_coords

    def get_problem_config(self) -> dict:
        """
        Return configuration dict for MFGProblem initialization.

        Returns:
            Dictionary with network-specific configuration
        """
        config = {
            "num_spatial_points": self.num_spatial_points,
            "spatial_shape": (self.num_spatial_points,),
            "spatial_discretization": None,
            "legacy_1d_attrs": None,
        }

        bounds = self.get_bounds()
        if bounds is not None:
            min_coords, max_coords = bounds
            config["spatial_bounds"] = tuple(
                (float(mn), float(mx)) for mn, mx in zip(min_coords, max_coords, strict=True)
            )
        else:
            config["spatial_bounds"] = None

        return config

    def is_on_boundary(
        self,
        points: np.ndarray,
        tolerance: float = ONWALL_TOL,
    ) -> np.ndarray:
        """
        Check if points are on the network boundary.

        For networks, boundary nodes are typically:
        - Leaf nodes (degree 1) for tree-like networks
        - Nodes on the spatial bounding box (if spatially embedded)

        Args:
            points: Array of shape (n, d) - points to check
            tolerance: Distance tolerance for boundary detection

        Returns:
            Boolean array of shape (n,) - True if point is on boundary
        """
        points = np.atleast_2d(points)
        bounds = self.get_bounds()

        if bounds is None:
            # Abstract network: no spatial boundary
            return np.zeros(len(points), dtype=bool)

        min_coords, max_coords = bounds
        on_boundary = np.zeros(len(points), dtype=bool)

        for i, p in enumerate(points):
            for d in range(len(min_coords)):
                if abs(p[d] - min_coords[d]) < tolerance or abs(p[d] - max_coords[d]) < tolerance:
                    on_boundary[i] = True
                    break

        return on_boundary

    def get_boundary_normal(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Get outward normal vectors at boundary points.

        For networks with spatial embedding, returns axis-aligned normals.

        Args:
            points: Array of shape (n, d) - boundary points

        Returns:
            Array of shape (n, d) - unit outward normal at each point
        """
        points = np.atleast_2d(points)
        bounds = self.get_bounds()

        if bounds is None:
            # Abstract network: return zero normals
            return np.zeros_like(points)

        min_coords, max_coords = bounds
        normals = np.zeros_like(points)
        tolerance = ONWALL_TOL

        for i, p in enumerate(points):
            for d in range(len(min_coords)):
                if abs(p[d] - min_coords[d]) < tolerance:
                    normals[i, d] = -1.0
                    break
                elif abs(p[d] - max_coords[d]) < tolerance:
                    normals[i, d] = 1.0
                    break

        return normals

    def project_to_boundary(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Project points onto the network boundary.

        For networks with spatial embedding, projects to nearest bounding box face.

        Args:
            points: Array of shape (n, d) - points to project

        Returns:
            Array of shape (n, d) - projected points on boundary
        """
        points = np.atleast_2d(points)
        bounds = self.get_bounds()

        if bounds is None:
            return points.copy()

        min_coords, max_coords = bounds
        projected = points.copy()

        for i, p in enumerate(points):
            min_dist = float("inf")
            best_proj = p.copy()

            for d in range(len(min_coords)):
                dist_min = abs(p[d] - min_coords[d])
                if dist_min < min_dist:
                    min_dist = dist_min
                    best_proj = p.copy()
                    best_proj[d] = min_coords[d]

                dist_max = abs(p[d] - max_coords[d])
                if dist_max < min_dist:
                    min_dist = dist_max
                    best_proj = p.copy()
                    best_proj[d] = max_coords[d]

            projected[i] = best_proj

        return projected

    def project_to_interior(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Project points outside the network domain back into the interior.

        For networks with spatial embedding, clips to bounding box.

        Args:
            points: Array of shape (n, d) - points to project

        Returns:
            Array of shape (n, d) - points guaranteed to be inside domain
        """
        points = np.atleast_2d(points)
        bounds = self.get_bounds()

        if bounds is None:
            return points.copy()

        min_coords, max_coords = bounds
        return np.clip(points, min_coords, max_coords)

    def get_boundary_regions(self) -> dict[str, dict]:
        """
        Get information about distinct boundary regions.

        For networks with spatial embedding, returns axis-aligned boundary regions.
        For abstract networks, returns empty dict.

        Returns:
            Dictionary mapping region names to region info
        """
        bounds = self.get_bounds()

        if bounds is None:
            return {}

        min_coords, max_coords = bounds
        regions = {}
        axis_names = ["x", "y", "z", "w", "v"]

        for d in range(len(min_coords)):
            axis = axis_names[d] if d < len(axis_names) else f"dim{d}"
            regions[f"{axis}_min"] = {"axis": d, "side": "min", "value": float(min_coords[d])}
            regions[f"{axis}_max"] = {"axis": d, "side": "max", "value": float(max_coords[d])}

        return regions


class GridNetwork(NetworkGeometry):
    """Grid/lattice network for MFG problems."""

    def __init__(
        self,
        width: int,
        height: int | None = None,
        periodic: bool = False,
        backend_preference: NetworkBackendType = NetworkBackendType.IGRAPH,
        boundary_conditions=None,
    ):
        self.width = width
        self.height = height or width
        self.periodic = periodic
        super().__init__(
            self.width * self.height,
            NetworkType.GRID,
            backend_preference,
            boundary_conditions=boundary_conditions,
        )
        # Issue #875: Auto-initialize network so users don't need to call create_network()
        self.create_network()

    def get_adjacency_matrix(self) -> NDArray:
        """Get adjacency matrix for the grid network."""
        if self.network_data is None:
            raise ValueError("Network data not initialized. Call create_network() first.")
        return self.network_data.adjacency_matrix.toarray()

    def create_network(self, **kwargs) -> NetworkData:
        """Create grid network using optimal backend."""
        nodes = self.width * self.height

        # Choose optimal backend for this size
        backend_manager = self.backend_manager
        backend_type = backend_manager.choose_backend(
            nodes, OperationType.GENERAL, force_backend=kwargs.get("force_backend")
        )
        backend = backend_manager.get_backend(backend_type)

        # Create graph using chosen backend
        graph = backend.create_graph(nodes, directed=False)

        # Generate node positions
        positions = np.zeros((nodes, 2))
        edges = []

        for i in range(self.height):
            for j in range(self.width):
                node_id = i * self.width + j
                positions[node_id] = [j, i]

                # Connect to neighbors
                neighbors = self._get_grid_neighbors(i, j)
                for ni, nj in neighbors:
                    neighbor_id = ni * self.width + nj
                    if node_id < neighbor_id:  # Avoid duplicate edges
                        edges.append((node_id, neighbor_id))

        # Add edges to backend graph
        if edges:
            backend.add_edges(graph, edges)

        # Convert to NetworkData
        self.network_data = self._create_network_data_from_backend(graph, backend_type, positions)

        return self.network_data

    def _get_grid_neighbors(self, i: int, j: int) -> list[tuple[int, int]]:
        """Get grid neighbors with optional periodic boundary conditions."""
        neighbors = []

        # 4-connected grid (can be extended to 8-connected)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for di, dj in directions:
            ni, nj = i + di, j + dj

            if self.periodic:
                ni = ni % self.height
                nj = nj % self.width
                neighbors.append((ni, nj))
            else:
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    neighbors.append((ni, nj))

        return neighbors

    def compute_distance_matrix(self) -> np.ndarray:
        """Compute Manhattan distances on grid."""
        distances = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                i1, j1 = divmod(i, self.width)
                i2, j2 = divmod(j, self.width)
                distances[i, j] = abs(i1 - i2) + abs(j1 - j2)

        return distances


class RandomNetwork(NetworkGeometry):
    """Random network (Erdos-Renyi model) for MFG problems."""

    def __init__(
        self,
        num_nodes: int,
        connection_prob: float = 0.1,
        backend_preference: NetworkBackendType = NetworkBackendType.IGRAPH,
    ):
        self.connection_prob = connection_prob
        super().__init__(num_nodes, NetworkType.RANDOM, backend_preference)
        # Issue #875: Auto-initialize network
        self.create_network()

    def get_adjacency_matrix(self) -> NDArray:
        """Get adjacency matrix for the random network."""
        if self.network_data is None:
            raise ValueError("Network data not initialized. Call create_network() first.")
        return self.network_data.adjacency_matrix.toarray()

    def create_network(self, seed: int | None = None, **kwargs) -> NetworkData:
        """Create random network using optimal backend."""
        if seed is not None:
            np.random.seed(seed)

        # Choose optimal backend for this size
        backend_manager = self.backend_manager
        backend_type = backend_manager.choose_backend(
            self.num_nodes,
            OperationType.GENERAL,
            force_backend=kwargs.get("force_backend"),
        )
        backend = backend_manager.get_backend(backend_type)

        # Create graph using chosen backend
        graph = backend.create_graph(self.num_nodes, directed=False)

        # Generate random edges
        edges = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if np.random.random() < self.connection_prob:
                    edges.append((i, j))

        # Add edges to backend graph
        if edges:
            backend.add_edges(graph, edges)

        # Generate random node positions for visualization
        positions = np.random.rand(self.num_nodes, 2)

        # Convert to NetworkData
        self.network_data = self._create_network_data_from_backend(graph, backend_type, positions)

        return self.network_data

    def compute_distance_matrix(self) -> np.ndarray:
        """Compute shortest path distances using BFS."""
        if not NETWORKX_AVAILABLE or nx is None:
            raise ImportError("NetworkX required for shortest path computation")

        if self.network_data is None:
            raise ValueError("Network data not initialized. Call create_network() first.")

        G = nx.from_scipy_sparse_array(self.network_data.adjacency_matrix)
        # Use floyd_warshall_numpy for distance matrix computation
        try:
            return np.array(nx.floyd_warshall_numpy(G))
        except AttributeError:
            # Fallback for older NetworkX versions
            paths = dict(nx.all_pairs_shortest_path_length(G))
            n = len(G.nodes())
            distance_matrix = np.full((n, n), np.inf)
            for i, i_paths in paths.items():
                for j, distance in i_paths.items():
                    distance_matrix[i, j] = distance
            return distance_matrix


class ScaleFreeNetwork(NetworkGeometry):
    """Scale-free network (Barabasi-Albert model) for MFG problems."""

    def __init__(
        self,
        num_nodes: int,
        num_edges_per_node: int = 2,
        backend_preference: NetworkBackendType = NetworkBackendType.IGRAPH,
    ):
        self.num_edges_per_node = num_edges_per_node
        super().__init__(num_nodes, NetworkType.SCALE_FREE, backend_preference)
        # Issue #875: Auto-initialize network
        self.create_network()

    def get_adjacency_matrix(self) -> NDArray:
        """Get adjacency matrix for the scale-free network."""
        if self.network_data is None:
            raise ValueError("Network data not initialized. Call create_network() first.")
        return self.network_data.adjacency_matrix.toarray()

    def create_network(self, seed: int | None = None, **kwargs) -> NetworkData:
        """Create scale-free network using optimal backend."""
        if seed is not None:
            np.random.seed(seed)

        # Choose optimal backend for this size
        backend_manager = self.backend_manager
        backend_type = backend_manager.choose_backend(
            self.num_nodes,
            OperationType.GENERAL,
            force_backend=kwargs.get("force_backend"),
        )
        backend = backend_manager.get_backend(backend_type)

        # Create graph using chosen backend
        graph = backend.create_graph(self.num_nodes, directed=False)

        # Generate Barabási-Albert network edges
        edges = self._generate_barabasi_albert_edges(seed)

        # Add edges to backend graph
        if edges:
            backend.add_edges(graph, edges)

        # Generate positions using spring-like layout
        positions = self._generate_spring_positions(seed)

        # Convert to NetworkData
        self.network_data = self._create_network_data_from_backend(graph, backend_type, positions)

        return self.network_data

    def _generate_barabasi_albert_edges(self, seed: int | None = None) -> list[tuple[int, int]]:
        """Generate Barabási-Albert network edges using preferential attachment."""
        if seed is not None:
            np.random.seed(seed)

        edges = []
        node_degrees = np.zeros(self.num_nodes)

        # Start with a complete graph of m+1 nodes
        initial_nodes = min(self.num_edges_per_node + 1, self.num_nodes)
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                edges.append((i, j))
                node_degrees[i] += 1
                node_degrees[j] += 1

        # Add remaining nodes with preferential attachment
        for new_node in range(initial_nodes, self.num_nodes):
            # Select m existing nodes to connect to based on their degrees
            targets: list[int] = []
            total_degree = np.sum(node_degrees[:new_node])

            if total_degree > 0:
                # Preferential attachment: probability proportional to degree
                probabilities = node_degrees[:new_node] / total_degree
                targets = np.random.choice(
                    new_node,
                    size=min(self.num_edges_per_node, new_node),
                    replace=False,
                    p=probabilities,
                ).tolist()
            else:
                # Fallback: random attachment
                targets = np.random.choice(
                    new_node, size=min(self.num_edges_per_node, new_node), replace=False
                ).tolist()

            # Add edges
            for target in targets:
                edges.append((new_node, target))
                node_degrees[new_node] += 1
                node_degrees[target] += 1

        return edges

    def _generate_spring_positions(self, seed: int | None = None) -> np.ndarray:
        """Generate node positions using spring-like layout."""
        if seed is not None:
            np.random.seed(seed)

        # Simple circular layout as fallback
        angles = np.linspace(0, 2 * np.pi, self.num_nodes, endpoint=False)
        radius = 1.0
        positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

        # Add some randomness
        positions += 0.1 * np.random.randn(self.num_nodes, 2)

        return positions

    def compute_distance_matrix(self) -> np.ndarray:
        """Compute shortest path distances."""
        if not NETWORKX_AVAILABLE or nx is None:
            raise ImportError("NetworkX required for shortest path computation")

        if self.network_data is None:
            raise ValueError("Network data not initialized. Call create_network() first.")

        G = nx.from_scipy_sparse_array(self.network_data.adjacency_matrix)
        # Use floyd_warshall_numpy for distance matrix computation
        try:
            return np.array(nx.floyd_warshall_numpy(G))
        except AttributeError:
            # Fallback for older NetworkX versions
            paths = dict(nx.all_pairs_shortest_path_length(G))
            n = len(G.nodes())
            distance_matrix = np.full((n, n), np.inf)
            for i, i_paths in paths.items():
                for j, distance in i_paths.items():
                    distance_matrix[i, j] = distance
            return distance_matrix


class CustomNetwork(NetworkGeometry):
    """
    Custom network from an existing networkx graph or adjacency matrix.

    This class wraps an existing graph structure to provide a NetworkGeometry
    interface, enabling integration with MFGProblem's geometry-first API.

    Example:
        >>> import networkx as nx
        >>> G = nx.grid_2d_graph(10, 10)
        >>> geometry = CustomNetwork.from_networkx(G)
        >>> problem = MFGProblem(geometry=geometry, T=1.0, Nt=50)
    """

    def __init__(
        self,
        adjacency_matrix: csr_matrix | np.ndarray,
        node_positions: np.ndarray | None = None,
        backend_preference: NetworkBackendType = NetworkBackendType.IGRAPH,
    ):
        """
        Create a custom network from an adjacency matrix.

        Args:
            adjacency_matrix: Sparse or dense adjacency matrix (N x N)
            node_positions: Optional (N, d) array of node coordinates
            backend_preference: Preferred backend for operations
        """
        # Convert to sparse if needed
        if isinstance(adjacency_matrix, np.ndarray):
            adjacency_matrix = csr_matrix(adjacency_matrix)

        num_nodes = adjacency_matrix.shape[0]
        super().__init__(num_nodes, NetworkType.CUSTOM, backend_preference)

        self._adjacency_matrix = adjacency_matrix
        self._node_positions = node_positions

        # Create network immediately
        self.create_network()

    @classmethod
    def from_networkx(
        cls,
        graph: nx.Graph,
        node_positions: np.ndarray | None = None,
        backend_preference: NetworkBackendType = NetworkBackendType.IGRAPH,
    ) -> CustomNetwork:
        """
        Create CustomNetwork from a NetworkX graph.

        Args:
            graph: NetworkX graph object
            node_positions: Optional node positions. If None and graph has 'pos'
                           attribute, those positions will be used.
            backend_preference: Preferred backend for operations

        Returns:
            CustomNetwork instance wrapping the graph
        """
        if not NETWORKX_AVAILABLE or nx is None:
            raise ImportError("NetworkX is required for from_networkx()")

        # Get adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(graph)

        # Try to get positions from graph if not provided
        if node_positions is None:
            pos_dict = nx.get_node_attributes(graph, "pos")
            if pos_dict:
                # Convert position dict to array
                nodes = list(graph.nodes())
                node_positions = np.array([pos_dict[n] for n in nodes])

        return cls(adjacency_matrix, node_positions, backend_preference)

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix as dense array."""
        return self._adjacency_matrix.toarray()

    def create_network(self, **kwargs) -> NetworkData:
        """Create NetworkData from the stored adjacency matrix."""
        # Count edges (for undirected: number of non-zero entries / 2)
        num_edges = self._adjacency_matrix.nnz // 2

        self.network_data = NetworkData(
            adjacency_matrix=self._adjacency_matrix,
            num_nodes=self._num_nodes,
            num_edges=num_edges,
            network_type=NetworkType.CUSTOM,
            node_positions=self._node_positions,
            backend_type=self.backend_preference,
        )

        return self.network_data

    def compute_distance_matrix(self) -> np.ndarray:
        """Compute shortest path distances using BFS."""
        if not NETWORKX_AVAILABLE or nx is None:
            raise ImportError("NetworkX required for shortest path computation")

        G = nx.from_scipy_sparse_array(self._adjacency_matrix)

        try:
            return np.array(nx.floyd_warshall_numpy(G))
        except AttributeError:
            # Fallback for older NetworkX versions
            paths = dict(nx.all_pairs_shortest_path_length(G))
            n = len(G.nodes())
            distance_matrix = np.full((n, n), np.inf)
            for i, i_paths in paths.items():
                for j, distance in i_paths.items():
                    distance_matrix[i, j] = distance
            return distance_matrix


# Factory function for creating networks
def create_network(
    network_type: str | NetworkType,
    num_nodes: int,
    backend_preference: NetworkBackendType = NetworkBackendType.IGRAPH,
    **kwargs,
) -> NetworkGeometry:
    """
    Factory function for creating network geometries with backend selection.

    Args:
        network_type: Type of network to create
        num_nodes: Number of nodes in network
        backend_preference: Preferred backend (igraph, networkit, networkx)
        **kwargs: Network-specific parameters

    Returns:
        Configured network geometry object with optimal backend
    """
    if isinstance(network_type, str):
        network_type = NetworkType(network_type)

    if network_type == NetworkType.GRID:
        width = kwargs.get("width", int(np.sqrt(num_nodes)))
        height = kwargs.get("height", num_nodes // width)
        periodic = kwargs.get("periodic", False)
        return GridNetwork(width, height, periodic, backend_preference)

    elif network_type == NetworkType.RANDOM:
        connection_prob = kwargs.get("connection_prob", 0.1)
        return RandomNetwork(num_nodes, connection_prob, backend_preference)

    elif network_type == NetworkType.SCALE_FREE:
        num_edges_per_node = kwargs.get("num_edges_per_node", 2)
        return ScaleFreeNetwork(num_nodes, num_edges_per_node, backend_preference)

    elif network_type == NetworkType.CUSTOM:
        # CUSTOM type requires either a graph or adjacency_matrix in kwargs
        graph = kwargs.get("graph")
        adjacency_matrix = kwargs.get("adjacency_matrix")
        node_positions = kwargs.get("node_positions")

        if graph is not None:
            # Create from networkx graph (use keyword args to avoid position confusion)
            return CustomNetwork.from_networkx(
                graph,
                node_positions=node_positions,
                backend_preference=backend_preference,
            )
        elif adjacency_matrix is not None:
            # Create from adjacency matrix
            return CustomNetwork(
                adjacency_matrix,
                node_positions=node_positions,
                backend_preference=backend_preference,
            )
        else:
            raise ValueError(
                "NetworkType.CUSTOM requires either 'graph' (networkx Graph) or 'adjacency_matrix' in kwargs"
            )

    else:
        raise ValueError(f"Unsupported network type: {network_type}")


# Utility functions for network analysis
def compute_network_statistics(network_data: NetworkData) -> dict[str, float]:
    """Compute comprehensive network statistics."""
    stats = {
        "num_nodes": network_data.num_nodes,
        "num_edges": network_data.num_edges,
        "density": network_data.metadata.get("density", 0),
        "average_degree": network_data.metadata.get("average_degree", 0),
        "max_degree": network_data.metadata.get("max_degree", 0),
        "clustering_coefficient": network_data.metadata.get("clustering_coefficient", 0),
        "is_connected": network_data.is_connected,
        "is_directed": network_data.is_directed,
        "is_weighted": network_data.is_weighted,
    }

    # Add spectral properties if possible
    try:
        if network_data.laplacian_matrix is not None:
            L = network_data.laplacian_matrix.toarray()
            eigenvals = np.linalg.eigvals(L)
            eigenvals = np.sort(eigenvals)

            stats.update(
                {
                    "algebraic_connectivity": eigenvals[1] if len(eigenvals) > 1 else 0,
                    "spectral_gap": (eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0),
                    "largest_eigenvalue": eigenvals[-1],
                }
            )
    except (ValueError, np.linalg.LinAlgError, MemoryError) as e:
        # Issue #547: Replace silent fallback with logged warning
        # Spectral analysis can fail for large graphs or numerical issues
        logger.warning(
            "Spectral analysis failed for network with %d nodes: %s. "
            "Spectral properties will be omitted from statistics.",
            network_data.num_nodes,
            e,
        )
        stats["spectral_analysis_failed"] = True

    return stats
