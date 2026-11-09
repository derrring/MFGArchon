"""
Graph-specific trait protocols for discrete geometry capabilities.

This module defines Protocol classes for graph geometry capabilities that parallel
the continuous geometry traits but reflect the discrete nature of graphs:

- Graph Laplacian (L = D - A) for discrete diffusion
- Adjacency matrix for connectivity
- Spatial embedding for positioned graphs
- Graph-theoretic distances

These protocols enable trait-based solver design for MFG on networks while
maintaining consistency with the continuous geometry trait system.

Example:
    def solve_network_mfg(geometry: SupportsGraphLaplacian):
        L = geometry.get_graph_laplacian_operator()
        # Discrete diffusion: dm/dt = -Lm
        m_new = m - dt * (L @ m)

Design Principle:
    Graph traits use the same naming pattern (Supports*) as continuous traits,
    but with semantics appropriate for discrete topology.

Created: 2026-01-17 (Issue #590 - Phase 1.3)
Part of: Geometry & BC Architecture Implementation (Issue #589)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@runtime_checkable
class SupportsGraphLaplacian(Protocol):
    """
    Geometry can provide discrete graph Laplacian operator.

    Graph Laplacian encodes network connectivity and diffusion structure.
    Unlike continuous Laplacian Δ which operates on fields, graph Laplacian L
    operates on node values via matrix multiplication.

    Mathematical Definition:
        L = D - A  (unnormalized Laplacian)
        L_norm = I - D^(-1/2) A D^(-1/2)  (normalized Laplacian)

    where:
        - D = diag(∑ⱼ Aᵢⱼ) is the degree matrix
        - A is the adjacency matrix
        - I is the identity matrix

    Properties:
        - L is symmetric and positive semi-definite
        - Smallest eigenvalue is 0 (constant eigenvector)
        - Spectrum encodes graph structure (spectral clustering)

    Used For:
        - Discrete diffusion: ∂m/∂t = -σ²Lm
        - Random walks on graphs
        - Spectral graph theory
        - Graph-based PDEs

    Example:
        >>> network = NetworkGeometry(adjacency_matrix=A)
        >>> L = network.get_graph_laplacian_operator(normalized=False)
        >>> # Discrete heat equation on graph
        >>> m_new = m - dt * sigma**2 * (L @ m)
    """

    def get_graph_laplacian_operator(
        self,
        normalized: bool = False,
    ) -> NDArray | Callable[[NDArray], NDArray]:
        """
        Return discrete graph Laplacian.

        Args:
            normalized: If True, return normalized Laplacian L_norm = I - D^(-1/2) A D^(-1/2)
                       If False, return unnormalized L = D - A (default)

        Returns:
            Laplacian matrix L of shape (N, N) or LinearOperator wrapping it

        Note:
            - Normalized Laplacian has eigenvalues in [0, 2]
            - Unnormalized Laplacian eigenvalues depend on graph degree
            - For solvers: use unnormalized for diffusion, normalized for spectral analysis

        Example:
            >>> L = network.get_graph_laplacian_operator(normalized=False)
            >>> eigenvalues = np.linalg.eigvalsh(L)
            >>> # Second smallest eigenvalue (Fiedler value) measures connectivity
        """
        ...


@runtime_checkable
class SupportsAdjacency(Protocol):
    """
    Geometry can provide graph adjacency information.

    Adjacency matrix A encodes the edge structure of the graph:
        - A[i,j] > 0: Edge exists from node i to node j with weight A[i,j]
        - A[i,j] = 0: No edge between nodes i and j

    For unweighted graphs, use A[i,j] ∈ {0, 1}.
    For directed graphs, A may be asymmetric.

    Used For:
        - Agent movement constraints (can only move along edges)
        - Pathfinding and routing
        - Network flow problems
        - Graph traversal algorithms

    Example:
        >>> network = NetworkGeometry(...)
        >>> A = network.get_adjacency_matrix()
        >>> # Check if node 5 connects to node 8
        >>> if A[5, 8] > 0:
        ...     print(f"Edge exists with weight {A[5, 8]}")
    """

    def get_adjacency_matrix(self) -> NDArray:
        """
        Return adjacency matrix for the graph.

        Returns:
            Adjacency matrix A of shape (N, N) where N is number of nodes
                - A[i,j] = weight of edge from node i to node j
                - A[i,j] = 0 if no edge exists
                - For unweighted graphs: A[i,j] ∈ {0, 1}
                - For undirected graphs: A = A^T (symmetric)

        Example:
            >>> A = network.get_adjacency_matrix()
            >>> # Find all neighbors of node 5
            >>> neighbors = np.nonzero(A[5])[0]
            >>> edge_weights = A[5, neighbors]
        """
        ...

    def get_neighbors(self, node_idx: int) -> list[int]:
        """
        Get neighbor indices for a node.

        Args:
            node_idx: Node index (0 ≤ node_idx < N)

        Returns:
            List of neighbor node indices (nodes connected by edges)

        Raises:
            IndexError: If node_idx out of range

        Example:
            >>> neighbors = network.get_neighbors(node_idx=5)
            >>> # [3, 7, 12] - nodes connected to node 5 by edges
        """
        ...


@runtime_checkable
class SupportsSpatialEmbedding(Protocol):
    """
    Geometry can provide spatial coordinates for graph nodes.

    Spatially-embedded graphs have nodes positioned in Euclidean space ℝ^d.
    This enables geometric computations while preserving graph topology.

    Examples of Spatially-Embedded Graphs:
        - Road networks (nodes at geographic coordinates)
        - Grid graphs / mazes (nodes at (x,y) grid positions)
        - Sensor networks (nodes at physical sensor locations)
        - Neural networks (neurons with spatial positions)

    Abstract graphs (social networks, dependency graphs) have NO spatial
    embedding and should not implement this protocol.

    Used For:
        - Euclidean distance computations
        - Visualization and rendering
        - Spatial locality (prefer nearby nodes in routing)
        - Hybrid discrete-continuous models

    Example:
        >>> maze = MazeGeometry(grid=walls_array)
        >>> positions = maze.get_node_positions()
        >>> # array([[0, 0], [0, 1], [1, 0], ...])  # (x, y) coordinates
        >>> dist = maze.get_euclidean_distance(node_i=0, node_j=5)
        >>> # 2.236... (Euclidean distance, not graph distance)
    """

    def get_node_positions(self) -> NDArray:
        """
        Get physical coordinates for nodes in embedding space.

        Returns:
            Node positions of shape (N, d) where:
                - N is number of nodes
                - d is embedding dimension (typically 2 or 3)
                - positions[i] = coordinates of node i in ℝ^d

        Raises:
            NotImplementedError: If graph has no spatial embedding

        Example:
            >>> positions = maze.get_node_positions()
            >>> # array([[0.0, 0.0],
            >>> #        [0.0, 1.0],
            >>> #        [1.0, 0.0],
            >>> #        ...])
            >>> x, y = positions[node_idx]
        """
        ...

    def get_euclidean_distance(
        self,
        node_i: int,
        node_j: int,
    ) -> float:
        """
        Compute Euclidean distance between nodes in embedding space.

        Args:
            node_i: First node index
            node_j: Second node index

        Returns:
            Euclidean distance |xᵢ - xⱼ| in embedding space

        Note:
            This is DIFFERENT from graph distance (shortest path length).
            For grid graphs: Euclidean distance ≤ graph distance (Manhattan/diagonal)

        Example:
            >>> # Nodes at (0,0) and (3,4)
            >>> dist = network.get_euclidean_distance(node_i=0, node_j=5)
            >>> # 5.0 (Euclidean distance √(3² + 4²))
            >>>
            >>> # But graph distance might be 7 (Manhattan path on grid)
            >>> graph_dist = network.get_graph_distance(0, 5, weighted=False)
            >>> # 7 hops
        """
        ...


@runtime_checkable
class SupportsGraphDistance(Protocol):
    """
    Geometry can compute graph-theoretic distances (shortest paths).

    Graph distance is the length of the shortest path between two nodes:
        - Unweighted: Number of hops (edges) in shortest path
        - Weighted: Sum of edge weights along shortest path

    This is fundamentally different from Euclidean distance for spatially-
    embedded graphs. Graph distance respects topology (can only travel along
    edges), while Euclidean distance is "as the crow flies".

    Used For:
        - Shortest path routing (Dijkstra, A*)
        - Network diameter (max distance between any two nodes)
        - Graph metrics and features
        - Distance-based kernels on graphs

    Example:
        >>> # Maze with obstacles
        >>> maze = MazeGeometry(...)
        >>> euclidean = maze.get_euclidean_distance(start, goal)  # 10.0 (straight line)
        >>> graph = maze.get_graph_distance(start, goal)  # 25.0 (path around walls)
    """

    def get_graph_distance(
        self,
        node_i: int,
        node_j: int,
        weighted: bool = False,
    ) -> float:
        """
        Compute graph distance (shortest path length) between nodes.

        Args:
            node_i: Source node index
            node_j: Target node index
            weighted: If True, use edge weights for path cost
                     If False, count hops (all edges have weight 1)

        Returns:
            Shortest path length from node_i to node_j
                - Unweighted: Integer number of hops
                - Weighted: Float sum of edge weights
                - Returns inf if no path exists

        Example:
            >>> # Unweighted: count edges in shortest path
            >>> hops = network.get_graph_distance(0, 10, weighted=False)
            >>> # 3 (path: 0 → 2 → 7 → 10, three edges)
            >>>
            >>> # Weighted: minimize total edge weight
            >>> cost = network.get_graph_distance(0, 10, weighted=True)
            >>> # 5.2 (path with minimum total weight)

        Note:
            - Implementation typically uses Dijkstra's algorithm (weighted)
              or BFS (unweighted)
            - For all-pairs distances, use Floyd-Warshall or precompute
            - Graph distance is a metric: d(i,j) ≥ 0, d(i,i) = 0, triangle inequality
        """
        ...

    def compute_all_pairs_distance(
        self,
        weighted: bool = False,
    ) -> NDArray:
        """
        Compute distance matrix for all node pairs.

        Args:
            weighted: If True, use edge weights. If False, count hops.

        Returns:
            Distance matrix D of shape (N, N) where D[i, j] = distance from i to j

        Example:
            >>> D = network.compute_all_pairs_distance(weighted=False)
            >>> # D[i, j] = number of hops from node i to node j
            >>> diameter = np.max(D[D < np.inf])  # Graph diameter
            >>> avg_path_length = np.mean(D[D < np.inf])  # Average path length

        Note:
            - This can be expensive: O(N³) for Floyd-Warshall
            - Consider caching if used repeatedly
            - D[i, j] = inf if no path exists (disconnected graph)
        """
        ...
