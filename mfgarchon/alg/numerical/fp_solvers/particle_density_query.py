"""
Particle Density Query for Efficient HJB-FP Coupling.

Provides spatial indexing and density estimation for particle-based Fokker-Planck
solvers, enabling efficient queries at specific points rather than full-grid KDE.

Performance Benefits:
    - Full-grid KDE: O(N_particles × N_grid) per timestep
    - Direct query: O(N_particles × log N_particles + N_query × log N_particles)
    - Speedup: 10-100× when N_query << N_grid

Applications:
    - Semi-Lagrangian HJB: Query density along characteristics
    - Policy iteration: Query at policy evaluation points
    - Adaptive mesh: Query only refined regions

Mathematical Background:
    Kernel Density Estimation at point x:
        m(x) = (1/N) ∑ᵢ K((x - Xᵢ)/h) / h^d

    k-NN Density Estimation:
        m(x) = k / (N · Volume(k-NN ball))

    Hybrid (kernel on k-NN subset):
        m(x) = ∑ᵢ∈kNN K((x - Xᵢ)/h) / (N · h^d)

Periodic Topology Support (Issue #714):
    For periodic (torus) domains, KDTree uses Euclidean distance which fails
    to find correct neighbors near domain boundaries. Solution: ghost particle
    augmentation - replicate particles near boundaries into shifted copies,
    then map neighbor indices back to original particles.

Boundary Correction Support (Issue #709):
    For reflecting/no-flux boundaries, standard KDE underestimates density at
    boundaries by ~50% because the kernel extends into "empty" space. Solution:
    reflection ghost particles - mirror particles about boundaries so the kernel
    "sees" reflected density, correcting the bias.

References:
    - Silverman (1986): Density Estimation for Statistics and Data Analysis
    - Scott (1992): Multivariate Density Estimation
    - scipy.spatial.KDTree documentation

Created: 2026-01-18 (Issue #489 Phase 1 - Core Query Infrastructure)
Updated: 2026-02-05 (Issue #714 - Periodic topology support)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.spatial import KDTree

from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


class ParticleDensityQuery:
    """
    Efficient particle density estimation using spatial indexing.

    Uses KD-tree for O(log N) nearest neighbor queries, enabling fast density
    estimation at arbitrary points without full-grid KDE computation.

    Examples
    --------
    >>> # Create query object from particles
    >>> particles = np.random.rand(10000, 2)  # 10k particles in 2D
    >>> query = ParticleDensityQuery(particles, bandwidth=0.1)
    >>>
    >>> # Query density at specific points
    >>> query_points = np.array([[0.5, 0.5], [0.3, 0.7]])
    >>> densities = query.query_density(query_points, method="kernel")
    >>>
    >>> # Batch query along characteristic
    >>> characteristics = np.linspace([0, 0], [1, 1], 50).reshape(-1, 2)
    >>> densities_path = query.query_density(characteristics)

    Notes
    -----
    **Method Comparison**:

    - "kernel": Gaussian kernel over all particles
      - Smooth, matches full KDE exactly
      - Cost: O(N_particles) per query

    - "knn": k-nearest neighbors
      - Fast, adaptive bandwidth
      - Cost: O(log N_particles) per query
      - Can produce discontinuities

    - "hybrid": Kernel over k-NN subset
      - Balanced: smooth + fast
      - Cost: O(k × log N_particles) per query
      - Recommended for most applications

    **Bandwidth Selection**:

    - Fixed: User-specified h
    - Scott's rule: h = N^(-1/(d+4)) × std(X)
    - Silverman's rule: h = (4/(d+2))^(1/(d+4)) × N^(-1/(d+4)) × std(X)
    """

    def __init__(
        self,
        particles: NDArray[np.float64],
        bandwidth: float | None = None,
        bandwidth_rule: Literal["fixed", "scott", "silverman"] = "fixed",
        periodic_bounds: list[tuple[float, float]] | None = None,
        periodic_dims: tuple[int, ...] | None = None,
        reflect_bounds: list[tuple[float, float]] | None = None,
        reflect_dims: tuple[int, ...] | None = None,
        reflect_margin: float | None = None,
    ):
        """
        Initialize particle density query with spatial index.

        Parameters
        ----------
        particles : NDArray
            Particle positions, shape (N_particles, dimension).
        bandwidth : float, optional
            Kernel bandwidth. If None, computed from bandwidth_rule.
        bandwidth_rule : str, default="fixed"
            Bandwidth selection rule:
            - "fixed": Use provided bandwidth (required if bandwidth is None)
            - "scott": Scott's rule of thumb
            - "silverman": Silverman's rule of thumb
        periodic_bounds : list of (float, float), optional
            Domain bounds for periodic topology, e.g. [(0, 1), (0, 1)].
            If provided with periodic_dims, enables ghost particle augmentation
            for correct neighbor search across periodic boundaries (Issue #714).
        periodic_dims : tuple of int, optional
            Dimensions with periodic topology. If None and periodic_bounds is
            provided, all dimensions are treated as periodic.
            Example: (0,) for cylinder (periodic in x only).
        reflect_bounds : list of (float, float), optional
            Domain bounds for reflecting boundaries (no-flux/Neumann BC).
            Enables reflection ghost particles to correct boundary bias in KDE
            (Issue #709). Cannot be used together with periodic_bounds.
        reflect_dims : tuple of int, optional
            Dimensions with reflecting boundaries. If None and reflect_bounds
            is provided, all dimensions have reflecting boundaries.
        reflect_margin : float, optional
            Distance from boundary to include particles for reflection.
            Default: 3 * bandwidth. Particles within this margin of a boundary
            are reflected about that boundary.

        Raises
        ------
        ValueError
            If bandwidth is None and bandwidth_rule is "fixed".
            If both periodic_bounds and reflect_bounds are provided.
        """
        self.particles = particles
        self.N_particles, self.dimension = particles.shape
        self.periodic_bounds = periodic_bounds
        self.periodic_dims = periodic_dims
        self.reflect_bounds = reflect_bounds
        self.reflect_dims = reflect_dims

        # Validate: can't have both periodic and reflecting
        if periodic_bounds is not None and reflect_bounds is not None:
            raise ValueError(
                "Cannot specify both periodic_bounds and reflect_bounds. "
                "Use periodic for torus topology, reflect for no-flux boundaries."
            )

        # Bandwidth selection (needed before reflection for margin calculation)
        if bandwidth is None:
            if bandwidth_rule == "fixed":
                raise ValueError("bandwidth must be provided when bandwidth_rule='fixed'")
            self.bandwidth = self._compute_bandwidth(bandwidth_rule)
            logger.debug(f"Auto-selected bandwidth: h={self.bandwidth:.4f} ({bandwidth_rule} rule)")
        else:
            self.bandwidth = bandwidth
            logger.debug(f"Using fixed bandwidth: h={bandwidth:.4f}")

        # Issue #714: Handle periodic topology via ghost particle augmentation
        if periodic_bounds is not None:
            from mfgarchon.geometry.boundary.periodic import create_periodic_ghost_points

            # Default: all dimensions periodic if bounds provided but dims not specified
            if periodic_dims is None:
                periodic_dims = tuple(range(self.dimension))
                self.periodic_dims = periodic_dims

            # Create augmented point cloud with ghost copies
            self._particles_augmented, self._original_indices = create_periodic_ghost_points(
                particles, periodic_bounds, periodic_dims
            )
            # Build KD-tree on augmented points
            self.tree = KDTree(self._particles_augmented)
            n_ghosts = len(self._particles_augmented) - self.N_particles
            logger.debug(
                f"Built periodic KD-tree: {self.N_particles} particles + "
                f"{n_ghosts} ghosts = {len(self._particles_augmented)} total"
            )

        # Issue #709: Handle reflecting boundaries via reflection ghost particles
        elif reflect_bounds is not None:
            from mfgarchon.geometry.boundary.periodic import create_reflection_ghost_points

            # Default: all dimensions reflecting if bounds provided but dims not specified
            if reflect_dims is None:
                reflect_dims = tuple(range(self.dimension))
                self.reflect_dims = reflect_dims

            # Default margin: 3 * bandwidth (captures 99.7% of Gaussian kernel)
            margin = reflect_margin if reflect_margin is not None else 3.0 * self.bandwidth

            # Create augmented point cloud with reflected copies near boundaries
            self._particles_augmented, self._original_indices = create_reflection_ghost_points(
                particles, reflect_bounds, margin, reflect_dims
            )
            # Build KD-tree on augmented points
            self.tree = KDTree(self._particles_augmented)
            n_ghosts = len(self._particles_augmented) - self.N_particles
            logger.debug(
                f"Built reflecting KD-tree: {self.N_particles} particles + "
                f"{n_ghosts} ghosts = {len(self._particles_augmented)} total "
                f"(margin={margin:.4f})"
            )

        else:
            # Standard case (no periodic or reflecting boundaries)
            self._particles_augmented = particles
            self._original_indices = np.arange(self.N_particles, dtype=np.int64)
            self.tree = KDTree(particles)
            logger.debug(f"Built KD-tree for {self.N_particles} particles in {self.dimension}D")

    def _compute_bandwidth(self, rule: str) -> float:
        """
        Compute bandwidth using rule of thumb.

        Parameters
        ----------
        rule : str
            Bandwidth selection rule ("scott" or "silverman").

        Returns
        -------
        h : float
            Computed bandwidth.
        """
        std = np.std(self.particles, axis=0).mean()  # Average std across dimensions

        if rule == "scott":
            # Scott's rule: h = N^(-1/(d+4)) × std
            h = self.N_particles ** (-1.0 / (self.dimension + 4)) * std
        elif rule == "silverman":
            # Silverman's rule: h = (4/(d+2))^(1/(d+4)) × N^(-1/(d+4)) × std
            factor = (4.0 / (self.dimension + 2)) ** (1.0 / (self.dimension + 4))
            h = factor * self.N_particles ** (-1.0 / (self.dimension + 4)) * std
        else:
            raise ValueError(f"Unknown bandwidth rule: {rule}")

        return h

    def query_density(
        self,
        query_points: NDArray[np.float64],
        method: Literal["kernel", "knn", "hybrid"] = "hybrid",
        k: int = 50,
    ) -> NDArray[np.float64]:
        """
        Estimate density at query points.

        Parameters
        ----------
        query_points : NDArray
            Query point locations, shape (N_query, dimension) or (dimension,) for single point.
        method : str, default="hybrid"
            Density estimation method:
            - "kernel": Gaussian kernel over all particles (exact, slow)
            - "knn": k-nearest neighbors (fast, discontinuous)
            - "hybrid": Kernel over k-NN subset (recommended)
        k : int, default=50
            Number of nearest neighbors for "knn" and "hybrid" methods.

        Returns
        -------
        densities : NDArray
            Estimated density at each query point, shape (N_query,).

        Examples
        --------
        >>> # Single point query
        >>> density = query.query_density(np.array([0.5, 0.5]))
        >>>
        >>> # Batch query
        >>> points = np.random.rand(100, 2)
        >>> densities = query.query_density(points, method="hybrid", k=30)
        """
        # Handle single point query
        query_points = np.atleast_2d(query_points)

        if query_points.shape[1] != self.dimension:
            raise ValueError(f"Query points dimension {query_points.shape[1]} != particle dimension {self.dimension}")

        if method == "kernel":
            return self._query_kernel(query_points)
        elif method == "knn":
            return self._query_knn(query_points, k)
        elif method == "hybrid":
            return self._query_hybrid(query_points, k)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _query_kernel(self, query_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Full kernel density estimation (exact but slow).

        Evaluates Gaussian kernel at all particles for each query point.
        Cost: O(N_query × N_particles)

        Issue #714: For periodic domains, we compute distances to AUGMENTED
        particles (including ghosts) to capture periodic wraparound contributions.
        """
        N_query = query_points.shape[0]
        densities = np.zeros(N_query)

        h = self.bandwidth
        normalization = 1.0 / (self.N_particles * h**self.dimension)

        for i, x in enumerate(query_points):
            # Compute distances to all particles (augmented if periodic)
            distances = np.linalg.norm(self._particles_augmented - x, axis=1)

            # Gaussian kernel: K(r) = exp(-r²/2)
            kernel_values = np.exp(-0.5 * (distances / h) ** 2)

            # Density estimate (normalize by ORIGINAL count, not augmented)
            densities[i] = normalization * np.sum(kernel_values)

        return densities

    def _query_knn(self, query_points: NDArray[np.float64], k: int) -> NDArray[np.float64]:
        """
        k-nearest neighbors density estimation.

        Density proportional to inverse volume containing k neighbors.
        Cost: O(N_query × log N_particles)

        Issue #714: For periodic domains, KDTree on augmented particles gives
        correct geodesic distances (torus metric) via ghost point copies.
        """
        N_query = query_points.shape[0]
        densities = np.zeros(N_query)

        for i, x in enumerate(query_points):
            # Find k nearest neighbors (in augmented cloud if periodic)
            distances, _ = self.tree.query(x, k=k)

            # Radius to k-th neighbor
            radius = distances[-1]

            # Volume of d-dimensional ball
            if self.dimension == 1:
                volume = 2 * radius
            elif self.dimension == 2:
                volume = np.pi * radius**2
            elif self.dimension == 3:
                volume = (4.0 / 3.0) * np.pi * radius**3
            else:
                # General formula: V_d = π^(d/2) / Γ(d/2 + 1) × r^d
                # Approximation for high dimensions
                from scipy.special import gamma

                volume = (np.pi ** (self.dimension / 2.0)) / gamma(self.dimension / 2.0 + 1) * radius**self.dimension

            # Density: k neighbors in volume (normalize by ORIGINAL count)
            densities[i] = k / (self.N_particles * volume) if volume > 1e-12 else 0.0

        return densities

    def _query_hybrid(self, query_points: NDArray[np.float64], k: int) -> NDArray[np.float64]:
        """
        Hybrid: Gaussian kernel over k-nearest neighbors.

        Balances smoothness (kernel) with efficiency (k-NN subset).
        Cost: O(N_query × (log N_particles + k))

        Issue #714: For periodic domains, KDTree is built on augmented points
        (original + ghosts). Distances are correct because ghosts are physical
        copies. We use distances directly - no index mapping needed for kernel.
        """
        N_query = query_points.shape[0]
        densities = np.zeros(N_query)

        h = self.bandwidth
        normalization = 1.0 / (self.N_particles * h**self.dimension)

        for i, x in enumerate(query_points):
            # Find k nearest neighbors (in augmented cloud if periodic)
            distances, _indices = self.tree.query(x, k=k)

            # Gaussian kernel on nearby particles only
            # Note: distances are correct even for periodic case because
            # ghost particles are actual copies at shifted positions
            kernel_values = np.exp(-0.5 * (distances / h) ** 2)

            # Density estimate (scaled by ORIGINAL population, not augmented)
            densities[i] = normalization * np.sum(kernel_values)

        return densities
