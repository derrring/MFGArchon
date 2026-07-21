"""
Hyperrectangle Domain (Axis-Aligned Box in n-D)

Hyperrectangle: Cartesian product of intervals in each dimension
    D = [a₁, b₁] × [a₂, b₂] × ... × [aₐ, bₐ]

Most common domain type in MFG applications:
- 2D: Rectangle (e.g., [0,1] × [0,1] for unit square)
- 3D: Box (e.g., [0,1]³ for unit cube)
- 4D: Hypercube (e.g., [0,1]⁴ for unit tesseract)
- nD: General hyperrectangle

Advantages:
- Exact signed distance function (O(d) complexity)
- Efficient sampling (no rejection needed!)
- Natural for grid-based problems
- Easy periodic boundary conditions

References:
- TECHNICAL_REFERENCE_HIGH_DIMENSIONAL_MFG.md Section 4.1
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from mfgarchon.utils.deprecation import deprecated

from .implicit_domain import ImplicitDomain

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.geometry.collocation import CollocationPointSet


class Hyperrectangle(ImplicitDomain):
    """
    Axis-aligned hyperrectangle in n dimensions.

    Domain: D = [a₁, b₁] × [a₂, b₂] × ... × [aₐ, bₐ]

    Signed distance function (exact):
        φ(x) = max(max(a - x), max(x - b))

    where a = (a₁, ..., aₐ) and b = (b₁, ..., bₐ).

    Attributes:
        bounds: Array of shape (d, 2) where bounds[i] = [min_i, max_i]
        dimension: Spatial dimension

    Example:
        >>> # 2D unit square
        >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        >>> domain.contains(np.array([0.5, 0.5]))  # True

        >>> # 4D unit hypercube
        >>> domain_4d = Hyperrectangle(np.array([[0, 1]] * 4))
        >>> particles = domain_4d.sample_uniform(10000)
        >>> assert particles.shape == (10000, 4)
    """

    def __init__(
        self,
        bounds: NDArray[np.float64],
        periodic_dims: tuple[int, ...] | None = None,
    ) -> None:
        """
        Initialize hyperrectangle domain.

        Args:
            bounds: Array of shape (d, 2) where bounds[i] = [min_i, max_i]
            periodic_dims: Dimensions with periodic topology (for torus).
                E.g., (0, 1) for 2D torus, None for bounded domain.
                Issue #711: Periodic support for GFDM.

        Raises:
            ValueError: If bounds are invalid (min >= max)

        Example:
            >>> # 2D rectangle [0,2] × [0,1]
            >>> domain = Hyperrectangle(np.array([[0, 2], [0, 1]]))

            >>> # 2D torus (periodic in both dimensions)
            >>> torus = Hyperrectangle(np.array([[0, 1], [0, 1]]), periodic_dims=(0, 1))

            >>> # 3D box [-1,1]³
            >>> domain_3d = Hyperrectangle(np.array([[-1, 1]] * 3))
        """
        bounds = np.asarray(bounds, dtype=float)

        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}")

        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("bounds must have min < max for all dimensions")

        self.bounds = bounds
        self._dimension = bounds.shape[0]
        self._periodic_dims = tuple(periodic_dims) if periodic_dims else ()

    @property
    def dimension(self) -> int:
        """Spatial dimension of the hyperrectangle."""
        return self._dimension

    # =========================================================================
    # SupportsPeriodic Protocol Implementation (Issue #711)
    # =========================================================================

    @property
    def periodic_dimensions(self) -> tuple[int, ...]:
        """Get dimensions with periodic topology."""
        return self._periodic_dims

    def get_periods(self) -> dict[int, float]:
        """Get period lengths for periodic dimensions."""
        return {dim: self.bounds[dim, 1] - self.bounds[dim, 0] for dim in self._periodic_dims}

    def wrap_coordinates(self, points: NDArray) -> NDArray:
        """
        Wrap coordinates to canonical fundamental domain.

        Delegates to canonical wrap_positions() utility (DRY principle).
        Only wraps periodic dimensions, non-periodic dims unchanged.

        Args:
            points: Points to wrap, shape (num_points, dimension) or (dimension,)

        Returns:
            Wrapped coordinates in [xmin, xmax), same shape as input
        """
        if not self._periodic_dims:
            return points

        # Delegate to canonical utility (Issue #711: DRY)
        # Use boundary.periodic (parallel to enforcement.py for DIRICHLET/NEUMANN)
        from mfgarchon.geometry.boundary.periodic import wrap_positions

        bounds_list = [(self.bounds[d, 0], self.bounds[d, 1]) for d in range(self._dimension)]
        return wrap_positions(points, bounds_list, periodic_dims=self._periodic_dims)

    def compute_periodic_distance(
        self,
        points1: NDArray,
        points2: NDArray,
    ) -> NDArray:
        """
        Compute distance accounting for periodic topology.

        For periodic dim i: d_i = min(|Δx_i|, L_i - |Δx_i|)
        Total distance: d = sqrt(∑ d_i²)
        """
        if not self._periodic_dims:
            diff = points1 - points2
            return np.linalg.norm(diff, axis=-1)

        single_point = points1.ndim == 1
        if single_point:
            points1 = points1.reshape(1, -1)
            points2 = points2.reshape(1, -1)

        diff = points1 - points2
        diff_squared = diff**2

        for dim_idx in self._periodic_dims:
            L = self.bounds[dim_idx, 1] - self.bounds[dim_idx, 0]
            abs_diff = np.abs(diff[:, dim_idx])
            wrapped_diff = np.minimum(abs_diff, L - abs_diff)
            diff_squared[:, dim_idx] = wrapped_diff**2

        distances = np.sqrt(np.sum(diff_squared, axis=1))
        return distances[0] if single_point else distances

    def wrap_displacement(self, delta: NDArray) -> NDArray:
        """
        Wrap displacement vector to [-L/2, L/2] for periodic dimensions.

        For periodic dimension i:
            delta_wrapped[i] = delta[i] - L[i] * round(delta[i] / L[i])

        Args:
            delta: Displacement vectors, shape (num_points, dimension) or (dimension,)

        Returns:
            Wrapped displacement in [-L/2, L/2], same shape as input
        """
        if not self._periodic_dims:
            return delta

        single_point = delta.ndim == 1
        if single_point:
            delta = delta.reshape(1, -1)

        wrapped = delta.copy()
        for dim_idx in self._periodic_dims:
            L = self.bounds[dim_idx, 1] - self.bounds[dim_idx, 0]
            wrapped[:, dim_idx] = wrapped[:, dim_idx] - L * np.round(wrapped[:, dim_idx] / L)

        return wrapped[0] if single_point else wrapped

    def signed_distance(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        """
        Compute the signed distance to the hyperrectangle.

        For a point x and hyperrectangle [a, b]:
            φ(x) = max(max(a - x), max(x - b))

        Exact Euclidean distance in the interior and on the boundary. In the
        exterior it is the per-axis (Chebyshev) form, a strict lower bound off the
        face normals: at (2, 2) for the unit box it returns 1.0 where the true
        distance is √2 (probed; likewise 3.0 vs 3.606 at (3, 4)). Directly off a
        face it is exact -- (2, 0.5) gives 1.0, which is correct.

        Sign and the zero level set are exact everywhere, so level-set geometry,
        inside/outside tests and boundary detection are unaffected; only the
        exterior *magnitude* is approximate. Where an exact exterior distance is
        needed, use ``np.linalg.norm(x - project_to_boundary(x))``.

        Args:
            x: Point(s) - shape (d,) or (N, d)

        Returns:
            Signed distance(s) - scalar or shape (N,)

        Complexity:
            O(d) per point

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
            >>> domain.signed_distance(np.array([0.5, 0.5]))  # -0.5 (inside)
            >>> domain.signed_distance(np.array([1.5, 0.5]))  #  0.5 (outside)
            >>> domain.signed_distance(np.array([1.0, 0.5]))  #  0.0 (boundary)
        """
        x = np.asarray(x, dtype=float)
        is_single = x.ndim == 1

        if is_single:
            x = x.reshape(1, -1)

        if x.shape[1] != self.dimension:
            raise ValueError(f"Point dimension {x.shape[1]} does not match domain dimension {self.dimension}")

        # Distance to lower bounds: a - x (positive if x < a)
        lower_dist = self.bounds[:, 0] - x

        # Distance to upper bounds: x - b (positive if x > b)
        upper_dist = x - self.bounds[:, 1]

        # Maximum distance along each dimension
        # If inside on dimension i: max(lower_dist[i], upper_dist[i]) < 0
        max_dist_per_dim = np.maximum(lower_dist, upper_dist)

        # Signed distance is max over all dimensions
        sd = np.max(max_dist_per_dim, axis=1)

        return sd[0] if is_single else sd

    def get_bounding_box(self) -> NDArray[np.float64]:
        """
        Get bounding box (returns self for hyperrectangle).

        Returns:
            bounds: Array of shape (d, 2) where bounds[i] = [min_i, max_i]
        """
        return self.bounds.copy()

    def sample_uniform(self, n_samples: int, max_attempts: int = 100, seed: int | None = None) -> NDArray[np.float64]:
        """
        Sample particles uniformly from hyperrectangle.

        Efficient implementation: no rejection sampling needed!

        Args:
            n_samples: Number of particles
            max_attempts: Unused (kept for interface compatibility)
            seed: Random seed

        Returns:
            particles: Array of shape (n_samples, d)

        Complexity:
            O(n_samples * d) - no rejection!

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
            >>> particles = domain.sample_uniform(1000, seed=42)
            >>> assert np.all(particles >= 0) and np.all(particles <= 1)
        """
        if seed is not None:
            np.random.seed(seed)

        # Sample uniformly from each dimension independently
        particles = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=(n_samples, self.dimension))

        return particles

    @deprecated(
        since="v0.12.0",
        replacement="Use MeshfreeApplicator from mfgarchon.geometry.boundary instead.",
    )
    def apply_boundary_conditions(
        self,
        particles: NDArray[np.float64],
        bc_type: Literal["reflecting", "absorbing", "periodic"] = "reflecting",
    ) -> NDArray[np.float64]:
        """
        Apply boundary conditions (specialized for hyperrectangle).

        .. deprecated:: 0.12.0
            Use :class:`MeshfreeApplicator` from ``mfgarchon.geometry.boundary`` instead.
            This provides a unified interface for all geometry types.

        Args:
            particles: Particle positions - shape (N, d)
            bc_type: Boundary condition type
                - "reflecting": Reflect particles at boundaries
                - "absorbing": Remove particles outside domain
                - "periodic": Wrap particles to opposite side

        Returns:
            Updated particle positions

        Example:
            >>> # Deprecated usage:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
            >>> particles_reflected = domain.apply_boundary_conditions(particles, "reflecting")
            >>>
            >>> # New preferred usage:
            >>> from mfgarchon.geometry.boundary import MeshfreeApplicator
            >>> applicator = MeshfreeApplicator(domain)
            >>> particles_reflected = applicator.apply_particle_bc(particles, "reflecting")
        """
        particles = particles.copy()

        if bc_type == "reflecting":
            # Reflect particles using modular fold reflection
            # This handles particles that travel multiple domain widths in one step
            for dim in range(self.dimension):
                xmin = self.bounds[dim, 0]
                xmax = self.bounds[dim, 1]
                Lx = xmax - xmin

                if Lx > 1e-14:
                    # Modular fold reflection: position bounces back and forth with period 2*Lx
                    shifted = particles[:, dim] - xmin
                    period = 2 * Lx
                    pos_in_period = shifted % period
                    in_second_half = pos_in_period > Lx
                    pos_in_period[in_second_half] = period - pos_in_period[in_second_half]
                    particles[:, dim] = xmin + pos_in_period

            return particles

        elif bc_type == "absorbing":
            # Remove particles outside domain
            inside = np.all(
                (particles >= self.bounds[:, 0]) & (particles <= self.bounds[:, 1]),
                axis=1,
            )
            return particles[inside]

        elif bc_type == "periodic":
            # Wrap particles to opposite side
            for dim in range(self.dimension):
                width = self.bounds[dim, 1] - self.bounds[dim, 0]

                # Particles below lower bound
                below = particles[:, dim] < self.bounds[dim, 0]
                particles[below, dim] += width

                # Particles above upper bound
                above = particles[:, dim] > self.bounds[dim, 1]
                particles[above, dim] -= width

            return particles

        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

    def compute_volume(self, n_monte_carlo: int = 100000) -> float:
        """
        Compute exact volume (no Monte Carlo needed for hyperrectangle).

        Volume = ∏ᵢ (bᵢ - aᵢ)

        Args:
            n_monte_carlo: Unused (kept for interface compatibility)

        Returns:
            Exact volume

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 2], [0, 3]]))
            >>> domain.compute_volume()  # 2 * 3 = 6
        """
        return np.prod(self.bounds[:, 1] - self.bounds[:, 0])

    def get_center(self) -> NDArray:
        """
        Get center point of hyperrectangle.

        Returns:
            center: Array of shape (d,)

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 2], [0, 4]]))
            >>> domain.get_center()  # array([1, 2])
        """
        return (self.bounds[:, 0] + self.bounds[:, 1]) / 2

    def get_vertices(self) -> NDArray:
        """
        Get all vertices (corners) of hyperrectangle.

        Returns:
            vertices: Array of shape (2^d, d) with all corners

        Warning:
            Grows exponentially with dimension! Use only for low dimensions.

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))  # 2D square
            >>> vertices = domain.get_vertices()
            >>> # Returns: [[0,0], [0,1], [1,0], [1,1]]
        """
        if self.dimension > 10:
            raise ValueError(
                f"get_vertices() with dimension={self.dimension} would create "
                f"2^{self.dimension} = {2**self.dimension} vertices. Use d≤10."
            )

        # Generate all combinations of (min, max) for each dimension
        from itertools import product

        vertices = []
        for combo in product(*[(0, 1)] * self.dimension):
            vertex = np.array([self.bounds[i, combo[i]] for i in range(self.dimension)])
            vertices.append(vertex)

        return np.array(vertices)

    # =========================================================================
    # GFDM/Meshfree Convenience Methods
    # =========================================================================

    def get_collocation_points(
        self,
        n_interior: int,
        n_boundary: int = 0,
        method: Literal["lloyd", "sobol", "poisson_disk", "uniform"] = "lloyd",
        seed: int | None = None,
        min_separation: float | None = None,
        boundary_margin: float | None = None,
        max_fill_distance: float | None = None,
        target_mesh_ratio: float | Literal["optimal"] | None = None,
    ) -> CollocationPointSet:
        """
        Generate collocation points for meshfree methods (GFDM, RBF, etc.).

        This is a convenience wrapper around CollocationSampler that provides
        a unified interface for generating well-distributed points in the domain.

        **Mesh Quality Control - Two Approaches:**

        1. **Fixed N (for EOC studies)**: Use `target_mesh_ratio` to optimize
           point distribution while keeping N = n_interior + n_boundary exact.

        2. **Hard constraints (N may vary)**: Use `min_separation` and/or
           `max_fill_distance` to enforce absolute bounds on h and q.

        Args:
            n_interior: Number of interior collocation points
            n_boundary: Number of boundary collocation points (0 for interior only)
            method: Sampling method:
                - "lloyd" (default): CVT/Lloyd relaxation for quasi-uniform points
                - "sobol": Quasi-random low-discrepancy sequence
                - "poisson_disk": Blue noise with minimum spacing
                - "uniform": Pure random sampling
            seed: Random seed for reproducibility
            min_separation: (May change N) Minimum distance between points.
                Points closer than this are removed.
            boundary_margin: Minimum distance from interior to boundary.
                Defaults to min_separation if not specified.
            max_fill_distance: (May change N) Maximum fill distance.
                Points are added in sparse regions to achieve this.
            target_mesh_ratio: (Preserves N) Target mesh ratio h/q. Can be:
                - float (e.g., 5.0): Stop when h/q <= target
                - 'optimal': Auto-converge until plateau or h/q < 2.0
                Optimizes distribution via Lloyd relaxation. RECOMMENDED for EOC.

        Returns:
            CollocationPointSet with points, boundary_indices, normals, is_boundary

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
            >>>
            >>> # For EOC studies: auto-optimize mesh quality (recommended)
            >>> coll = domain.get_collocation_points(
            ...     n_interior=64, n_boundary=32,
            ...     target_mesh_ratio='optimal',  # Auto-converge to best h/q
            ... )
            >>> assert len(coll.points) == 96  # N preserved
            >>>
            >>> # For adaptive refinement: hard constraints, N may vary
            >>> coll = domain.get_collocation_points(
            ...     n_interior=64, n_boundary=32,
            ...     min_separation=0.05,
            ...     max_fill_distance=0.15,
            ... )

        See Also:
            - mfgarchon.utils.numerical.compute_mesh_distances for quality verification
        """
        from mfgarchon.geometry.collocation import CollocationSampler

        sampler = CollocationSampler(self)
        return sampler.generate_collocation(
            n_interior=n_interior,
            n_boundary=n_boundary,
            interior_method=method,
            seed=seed,
            min_separation=min_separation,
            boundary_margin=boundary_margin,
            max_fill_distance=max_fill_distance,
            target_mesh_ratio=target_mesh_ratio,
        )

    def get_boundary_indices(
        self,
        points: NDArray[np.float64],
        tolerance: float = 1e-6,
    ) -> NDArray[np.int64]:
        """
        Get indices of points that lie on the domain boundary.

        Convenience method for GFDM solvers that need boundary point indices
        for applying boundary conditions.

        Args:
            points: Array of points, shape (N, d)
            tolerance: Distance threshold for boundary detection

        Returns:
            Array of indices where |signed_distance| < tolerance

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
            >>> points = np.array([[0.5, 0.5], [0.0, 0.5], [1.0, 1.0]])
            >>> boundary_idx = domain.get_boundary_indices(points)
            >>> # boundary_idx = [1, 2] (points on edges/corners)
        """
        sdf = self.signed_distance(points)
        return np.where(np.abs(sdf) < tolerance)[0].astype(np.int64)

    def get_boundary_normals(
        self,
        points: NDArray[np.float64],
        eps: float = 1e-6,
    ) -> NDArray[np.float64]:
        """
        Compute outward unit normals at boundary points.

        For axis-aligned hyperrectangles, normals are exactly ±1 along
        coordinate axes. This method uses the SDF gradient for generality.

        Args:
            points: Boundary points, shape (N, d)
            eps: Finite difference step for gradient computation

        Returns:
            Outward unit normals, shape (N, d)

        Note:
            For points not exactly on the boundary, returns the gradient
            of the signed distance function (points toward boundary).
        """
        # Numerical gradient of SDF
        N, d = points.shape
        grad = np.zeros((N, d))

        for dim in range(d):
            shift = np.zeros(d)
            shift[dim] = eps
            sdf_plus = self.signed_distance(points + shift)
            sdf_minus = self.signed_distance(points - shift)
            grad[:, dim] = (sdf_plus - sdf_minus) / (2 * eps)

        # Normalize
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return grad / norms

    def __repr__(self) -> str:
        """String representation."""
        bounds_str = ", ".join([f"[{self.bounds[i, 0]}, {self.bounds[i, 1]}]" for i in range(min(self.dimension, 3))])
        if self.dimension > 3:
            bounds_str += ", ..."
        return f"Hyperrectangle({bounds_str})"
