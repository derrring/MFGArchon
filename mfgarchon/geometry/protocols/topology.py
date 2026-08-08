"""
Topology trait protocols for geometry capabilities.

This module defines Protocol classes for topological geometry properties.
Geometries implement these protocols to advertise their topological structure:
- Manifold structure (smooth differential geometry)
- Lipschitz continuity (boundary regularity for well-posedness)
- Periodic topology (toroidal domains)

These protocols enable:
- Riemannian MFG on manifolds (geodesic Hamiltonians)
- Well-posedness validation (GKS/Lopatinskii-Shapiro conditions)
- Periodic BC automation (wrap-around connectivity)

Example:
    # Check if geometry is periodic
    if isinstance(geometry, SupportsPeriodic):
        periods = geometry.get_periods()
        wrapped_points = geometry.wrap_coordinates(points)

    # For manifold MFG
    if isinstance(geometry, SupportsManifold):
        metric = geometry.get_metric_tensor(points)
        # Use metric for geodesic Hamiltonian

Created: 2026-01-17 (Issue #590 - Phase 1.1)
Part of: Geometry & BC Architecture Implementation (Issue #589)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class SupportsManifold(Protocol):
    """
    Geometry has smooth manifold structure.

    A manifold is a topological space that locally resembles Euclidean space.
    For MFG on manifolds, we need:
    - Riemannian metric (for geodesic Hamiltonians)
    - Tangent spaces (for velocity vectors)
    - Connection (for gradient/divergence operators)

    Mathematical Definition:
        M is a d-dimensional smooth manifold with Riemannian metric g.
        At each point x ∈ M, tangent space T_x M ≅ ℝ^d.

    Use Cases:
        - MFG on sphere S² (crowd motion on Earth surface)
        - MFG on Lie groups SO(3) (orientation-valued agents)
        - MFG on curved spaces (general relativity, cosmology)

    Discretization-Specific Implementations:
        - TensorProductGrid: Flat manifold (g = identity)
        - ImplicitDomain: Embedded submanifold (induced metric)
        - UnstructuredMesh: Piecewise-linear manifold approximation
        - GraphGeometry: Discrete graph (not a manifold)

    Reference:
        - Lee, "Introduction to Smooth Manifolds"
        - Cardaliaguet & Achdou, "Mean Field Games on Networks"
    """

    @property
    def manifold_dimension(self) -> int:
        """
        Intrinsic dimension of the manifold.

        Returns:
            Manifold dimension d (may differ from ambient dimension).

        Example:
            >>> # Circle embedded in ℝ²
            >>> circle = ImplicitDomain(sdf=lambda x: np.linalg.norm(x) - 1.0)
            >>> assert circle.manifold_dimension == 1  # 1D manifold
            >>> assert circle.dimension == 2  # 2D ambient space

        Note:
            - For TensorProductGrid: manifold_dimension == dimension
            - For embedded surfaces: manifold_dimension < dimension
        """
        ...

    def get_metric_tensor(
        self,
        points: NDArray,
    ) -> NDArray:
        """
        Compute Riemannian metric tensor at given points.

        The metric tensor g defines inner products on tangent spaces:
            <v, w>_x = v^T g(x) w

        For geodesic Hamiltonian:
            H(x, p) = (1/2) p^T g^{-1}(x) p

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,)

        Returns:
            Metric tensor(s), shape:
                - Single point: (dimension, dimension)
                - Multiple points: (num_points, dimension, dimension)
            Each matrix is symmetric positive definite.

        Example:
            >>> # Flat Euclidean space
            >>> grid = TensorProductGrid(...)
            >>> g = grid.get_metric_tensor(np.array([0.5, 0.5]))
            >>> assert np.allclose(g, np.eye(2))  # Identity metric
            >>>
            >>> # Sphere with radius R
            >>> sphere = ImplicitDomain(sdf=lambda x: np.linalg.norm(x) - R)
            >>> g = sphere.get_metric_tensor(np.array([R, 0, 0]))
            >>> # g is induced metric from embedding

        Note:
            - Flat geometries: g = I (identity)
            - Curved geometries: g computed from embedding or parameterization
            - For numerical stability, condition number of g should be reasonable
        """
        ...

    def get_tangent_space_basis(
        self,
        points: NDArray,
    ) -> NDArray:
        """
        Compute orthonormal basis for tangent space at given points.

        At each point x ∈ M, returns orthonormal vectors {e₁, e₂, ..., e_d}
        spanning the tangent space T_x M.

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,)

        Returns:
            Tangent basis vectors, shape:
                - Single point: (manifold_dimension, dimension)
                - Multiple points: (num_points, manifold_dimension, dimension)
            Each set of vectors is orthonormal w.r.t. metric.

        Example:
            >>> # Circle in ℝ²
            >>> circle = ImplicitDomain(sdf=lambda x: np.linalg.norm(x) - 1.0)
            >>> point = np.array([1.0, 0.0])  # On circle
            >>> basis = circle.get_tangent_space_basis(point)
            >>> assert basis.shape == (1, 2)  # 1D tangent space in ℝ²
            >>> assert np.allclose(basis, [[0.0, 1.0]])  # Tangent direction

        Note:
            - For embedded manifolds: basis is orthogonal to normal
            - For TensorProductGrid: returns canonical basis
            - Basis is orthonormal w.r.t. metric tensor
        """
        ...

    def compute_christoffel_symbols(
        self,
        points: NDArray,
    ) -> NDArray:
        """
        Compute Christoffel symbols (connection coefficients) at given points.

        Christoffel symbols Γᵢⱼᵏ define the covariant derivative:
            ∇_v w = ∂w/∂v + Γ(v, w)

        Used for:
            - Geodesic equation: d²x/dt² + Γ(dx/dt, dx/dt) = 0
            - Covariant Laplacian (Laplace-Beltrami operator)
            - Parallel transport

        Args:
            points: Query points, shape (num_points, dimension) or (dimension,)

        Returns:
            Christoffel symbols, shape:
                - Single point: (dimension, dimension, dimension)
                - Multiple points: (num_points, dimension, dimension, dimension)
            Γᵢⱼᵏ = Γ[k, i, j] is symmetric in i,j.

        Raises:
            NotImplementedError: If geometry doesn't support Christoffel computation
                                 (e.g., GraphGeometry, piecewise-linear meshes)

        Example:
            >>> # Flat space (Euclidean)
            >>> grid = TensorProductGrid(...)
            >>> Gamma = grid.compute_christoffel_symbols(np.array([0.5, 0.5]))
            >>> assert np.allclose(Gamma, 0)  # All zero for flat metric

        Note:
            - Flat geometries: Γ = 0 (trivial connection)
            - Curved geometries: Γ computed from metric tensor derivatives
            - Formula: Γᵢⱼᵏ = (1/2) g^{kl} (∂gⱼₗ/∂xⁱ + ∂gᵢₗ/∂xʲ - ∂gᵢⱼ/∂xₗ)
        """
        ...


@runtime_checkable
class SupportsLipschitz(Protocol):
    """
    Geometry has Lipschitz-continuous boundary.

    A domain Ω has Lipschitz boundary if ∂Ω can be locally represented as
    the graph of a Lipschitz-continuous function. This regularity condition
    is essential for:
    - PDE well-posedness (trace theorems, integration by parts)
    - Stability analysis (GKS condition, Lopatinskii-Shapiro condition)
    - Numerical convergence guarantees

    Mathematical Definition:
        Domain Ω ⊂ ℝ^d has Lipschitz boundary ∂Ω if:
        ∀x ∈ ∂Ω, ∃ neighborhood U and Lipschitz function f:
            U ∩ ∂Ω = {y ∈ U : y_d = f(y₁, ..., y_{d-1})}

    Use Cases:
        - GKS stability validation (Issue #535)
        - BC well-posedness checking
        - Numerical error analysis
        - Mesh quality validation

    Discretization-Specific Implementations:
        - TensorProductGrid: Piecewise-smooth boundary (Lipschitz ✓)
        - ImplicitDomain: Depends on SDF regularity
        - UnstructuredMesh: Piecewise-linear boundary (Lipschitz ✓)
        - GraphGeometry: Not applicable (discrete graph)

    Reference:
        - Evans, "Partial Differential Equations", Chapter 5.5
        - Kreiss & Lorenz, "Initial-Boundary Value Problems" (GKS condition)
    """

    def get_lipschitz_constant(
        self,
        region: str | None = None,
    ) -> float:
        """
        Return Lipschitz constant for boundary representation.

        For boundary locally represented as y_d = f(y₁, ..., y_{d-1}),
        the Lipschitz constant L satisfies:
            |f(x) - f(y)| ≤ L |x - y|

        Args:
            region: Optional boundary region name (e.g., "x_min", "obstacle").
                    If None, return maximum L over entire boundary.

        Returns:
            Lipschitz constant L ≥ 0.
            Special values:
                - L = 0: Constant (degenerate)
                - L = ∞: Not Lipschitz (corners, cusps)

        Example:
            >>> # Rectangular domain (axis-aligned boundary)
            >>> grid = TensorProductGrid(bounds=[(0,1), (0,1)], ...)
            >>> L = grid.get_lipschitz_constant()
            >>> assert np.isclose(L, 0)  # Piecewise constant (axis-aligned)
            >>>
            >>> # Circle (smooth boundary)
            >>> circle = ImplicitDomain(sdf=lambda x: np.linalg.norm(x) - 1.0)
            >>> L = circle.get_lipschitz_constant()
            >>> assert np.isfinite(L)  # Smooth => Lipschitz

        Note:
            - Rectangular domains: L = 0 (piecewise constant boundary)
            - Smooth boundaries: L = max |∇f| (gradient bound)
            - Corners/edges: L may be large but finite
            - Cusps: L = ∞ (not Lipschitz)
        """
        ...

    def validate_lipschitz_regularity(
        self,
        tolerance: float = 1e-6,
    ) -> tuple[bool, str]:
        """
        Validate that boundary satisfies Lipschitz condition.

        Checks:
        1. Boundary representable as graph of Lipschitz function
        2. No cusps, self-intersections, or degeneracies
        3. Lipschitz constant is finite and reasonable

        Args:
            tolerance: Numerical tolerance for validation checks

        Returns:
            (is_valid, message):
                - is_valid: True if boundary is Lipschitz
                - message: Diagnostic message (empty if valid, error description if not)

        Example:
            >>> grid = TensorProductGrid(bounds=[(0,1), (0,1)], ...)
            >>> valid, msg = grid.validate_lipschitz_regularity()
            >>> assert valid and msg == ""
            >>>
            >>> # Geometry with cusp
            >>> cusp = ImplicitDomain(sdf=lambda x: x[1]**2 - x[0]**3)
            >>> valid, msg = cusp.validate_lipschitz_regularity()
            >>> assert not valid and "cusp" in msg.lower()

        Note:
            - Used internally by GKS stability checker (Issue #535)
            - Validation may be expensive for complex geometries
            - For known-good geometries (TensorProductGrid), return True immediately
        """
        ...


@runtime_checkable
class SupportsPeriodic(Protocol):
    """
    Geometry has periodic topology (toroidal domain).

    Periodic geometries identify opposite boundaries, creating wrap-around
    connectivity. Common in:
    - Traffic flow on circular roads
    - Crystalline structure (periodic lattices)
    - Spatially-periodic MFG (infinite repeating patterns)

    Mathematical Definition:
        Domain Ω with periodic dimensions P ⊆ {1, 2, ..., d}.
        For each i ∈ P, opposite boundaries are identified:
            x_i = a_i  ≡  x_i = b_i  (modulo period L_i = b_i - a_i)

    Topology:
        - 1D periodic: Circle S¹
        - 2D periodic (both dims): Torus T² = S¹ × S¹
        - Mixed: Cylinder S¹ × [0,1] (periodic in x, bounded in y)

    Use Cases:
        - Circular racing track (1D periodic)
        - Pac-Man domain (2D periodic)
        - Crystalline MFG (3D periodic lattice)

    Discretization-Specific Implementations:
        - TensorProductGrid: Trivial (set periodic_dims in constructor)
        - ImplicitDomain: Complex (requires topological identification)
        - UnstructuredMesh: Requires periodic node mapping
        - GraphGeometry: Cyclic graph structure
    """

    @property
    def periodic_dimensions(self) -> tuple[int, ...]:
        """
        Get dimensions with periodic topology.

        Returns:
            Tuple of dimension indices (0-indexed) that are periodic.
            Empty tuple if no periodicity.

        Example:
            >>> # Cylinder: periodic in x, bounded in y
            >>> grid = TensorProductGrid(
            ...     dimension=2,
            ...     bounds=[(0, 2*np.pi), (0, 1)],
            ...     periodic_dims=(0,),  # x is periodic
            ... )
            >>> assert grid.periodic_dimensions == (0,)
            >>>
            >>> # Torus: periodic in both x and y
            >>> torus = TensorProductGrid(
            ...     dimension=2,
            ...     bounds=[(0, 2*np.pi), (0, 2*np.pi)],
            ...     periodic_dims=(0, 1),
            ... )
            >>> assert torus.periodic_dimensions == (0, 1)

        Note:
            - Indexing: 0 = x, 1 = y, 2 = z, etc.
            - For non-periodic geometries: returns empty tuple ()
        """
        ...

    def get_periods(self) -> dict[int, float]:
        """
        Get period lengths for periodic dimensions.

        Returns:
            Dictionary mapping dimension index → period length.
            Only includes periodic dimensions.

        Example:
            >>> grid = TensorProductGrid(
            ...     dimension=2,
            ...     bounds=[(0, 2*np.pi), (0, 1)],
            ...     periodic_dims=(0,),
            ... )
            >>> periods = grid.get_periods()
            >>> assert periods == {0: 2*np.pi}

        Note:
            - Period length L_i = b_i - a_i for dimension i
            - For angle coordinates, typically L = 2π
            - For spatial periodicity, L is domain size
        """
        ...

    def wrap_coordinates(
        self,
        points: NDArray,
    ) -> NDArray:
        """
        Wrap coordinates to canonical fundamental domain.

        For periodic dimension i with period L_i, wraps coordinates to [a_i, b_i):
            x_i_wrapped = a_i + (x_i - a_i) mod L_i

        Args:
            points: Points to wrap, shape (num_points, dimension) or (dimension,)

        Returns:
            Wrapped coordinates in fundamental domain, same shape as input

        Example:
            >>> # 1D periodic on [0, 2π)
            >>> grid = TensorProductGrid(
            ...     dimension=1,
            ...     bounds=[(0, 2*np.pi)],
            ...     periodic_dims=(0,),
            ... )
            >>> # Point outside domain
            >>> x_outside = np.array([3*np.pi])
            >>> x_wrapped = grid.wrap_coordinates(x_outside)
            >>> assert np.allclose(x_wrapped, [np.pi])  # 3π mod 2π = π
            >>>
            >>> # Negative coordinate
            >>> x_neg = np.array([-np.pi/2])
            >>> x_wrapped = grid.wrap_coordinates(x_neg)
            >>> assert np.allclose(x_wrapped, [3*np.pi/2])

        Note:
            - Non-periodic dimensions: coordinates unchanged
            - Use for particle positions that crossed periodic boundary
            - Essential for semi-Lagrangian methods on periodic domains
        """
        ...

    def compute_periodic_distance(
        self,
        points1: NDArray,
        points2: NDArray,
    ) -> NDArray:
        """
        Compute distance accounting for periodic topology.

        For periodic dimensions, computes shortest distance considering
        wrap-around connections. For non-periodic dimensions, uses standard
        Euclidean distance.

        Args:
            points1: First set of points, shape (num_points, dimension) or (dimension,)
            points2: Second set of points, same shape as points1

        Returns:
            Distances, shape (num_points,) or scalar for single point

        Example:
            >>> # 1D periodic on [0, 1)
            >>> grid = TensorProductGrid(
            ...     dimension=1,
            ...     bounds=[(0, 1)],
            ...     periodic_dims=(0,),
            ... )
            >>> # Points near opposite boundaries
            >>> x1 = np.array([0.1])
            >>> x2 = np.array([0.9])
            >>> # Standard distance: |0.9 - 0.1| = 0.8
            >>> # Periodic distance (minimum image): |−0.8 − 1·round(−0.8/1)| = 0.2
            >>> dist = grid.compute_periodic_distance(x1, x2)
            >>> assert np.isclose(dist, 0.2)

        Note:
            - For periodic dim i, take the minimum image:
              d_i = |δ_i - L_i * round(δ_i / L_i)| where δ_i = x1_i - x2_i
            - Total distance: d = sqrt(∑_i d_i²)
            - Critical for nearest-neighbor queries on periodic domains

        Implementers: do not restate the rule -- call
        ``mfgarchon.geometry.boundary.periodic.wrap_displacement``, its single owner.
        ~~d_i = min(|δ_i|, L_i - |δ_i|)~~ stood here as the normative form until 2026-08-08
        and is wrong for |δ_i| > L_i: on a unit torus it gives 0.9 for a separation of 1.9
        (true answer 0.1) and 6.7 for 7.7, a distance larger than the domain. Both shipped
        implementers had transcribed it from this Note, which is how one rule became three
        (Issue #1853; owner established in #1841, #1847).
        """
        ...
