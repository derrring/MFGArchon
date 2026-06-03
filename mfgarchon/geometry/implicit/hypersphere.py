"""
Hypersphere Domain (Ball in n-D)

Hypersphere: All points within distance r from center c
    D = {x ∈ ℝ^d : ||x - c|| ≤ r}

Common uses:
- 2D: Circle (e.g., circular obstacles)
- 3D: Sphere (e.g., spherical regions)
- nD: Hypersphere (e.g., safe zones in high-D)

Advantages:
- Exact signed distance function (O(d) complexity)
- Rotationally symmetric
- Natural for radial problems
- Perfect for obstacles

References:
- TECHNICAL_REFERENCE_HIGH_DIMENSIONAL_MFG.md Section 4.2
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .implicit_domain import ImplicitDomain


class Hypersphere(ImplicitDomain):
    """
    Hypersphere (ball) in n dimensions.

    Domain: D = {x ∈ ℝ^d : ||x - c|| ≤ r}

    Signed distance function (exact):
        φ(x) = ||x - c|| - r

    where c is the center and r is the radius.

    Attributes:
        center: Center point - array of shape (d,)
        radius: Radius (positive scalar)
        dimension: Spatial dimension

    Example:
        >>> # 2D unit circle centered at origin
        >>> circle = Hypersphere(center=[0, 0], radius=1.0)
        >>> circle.contains([0.5, 0.5])  # True

        >>> # 4D hypersphere
        >>> sphere_4d = Hypersphere(center=[0]*4, radius=1.0)
        >>> particles = sphere_4d.sample_uniform(10000)
        >>> assert particles.shape == (10000, 4)
    """

    def __init__(self, center: NDArray[np.float64] | Sequence[float], radius: float) -> None:
        """
        Initialize hypersphere domain.

        Args:
            center: Center point - array-like of shape (d,)
            radius: Radius (must be positive)

        Raises:
            ValueError: If radius is not a positive, finite scalar

        Example:
            >>> # 2D circle at (1, 2) with radius 0.5
            >>> circle = Hypersphere(center=[1.0, 2.0], radius=0.5)

            >>> # 3D sphere at origin with radius 2
            >>> sphere = Hypersphere(center=[0, 0, 0], radius=2.0)
        """
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")

        # Issue #1077: radius=inf/nan slips past the `<= 0` check (inf/nan comparisons are
        # False), then yields an infinite bounding box -> np.random.uniform(-inf, inf) -> NaN
        # in rejection sampling. Fail fast.
        if not np.isfinite(self.radius):
            raise ValueError(f"Radius must be finite, got {radius}")

        if self.center.ndim != 1:
            raise ValueError(f"Center must be 1D array, got shape {self.center.shape}")

        self._dimension = len(self.center)

    @property
    def dimension(self) -> int:
        """Spatial dimension of the hypersphere."""
        return self._dimension

    def signed_distance(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        """
        Compute exact signed distance to hypersphere.

        For a point x, center c, and radius r:
            φ(x) = ||x - c|| - r

        Args:
            x: Point(s) - shape (d,) or (N, d)

        Returns:
            Signed distance(s) - scalar float or array of shape (N,)

        Complexity:
            O(d) per point

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> sphere.signed_distance([0, 0])       # -1.0 (center)
            >>> sphere.signed_distance([1, 0])       #  0.0 (boundary)
            >>> sphere.signed_distance([2, 0])       #  1.0 (outside)
            >>> sphere.signed_distance([0.5, 0])     # -0.5 (inside)
        """
        x = np.asarray(x, dtype=float)
        is_single = x.ndim == 1

        if is_single:
            x = x.reshape(1, -1)

        if x.shape[1] != self.dimension:
            raise ValueError(f"Point dimension {x.shape[1]} does not match domain dimension {self.dimension}")

        # Euclidean distance from center
        distances = np.linalg.norm(x - self.center, axis=1)

        # Signed distance: dist - radius
        sd = distances - self.radius

        return sd[0] if is_single else sd

    def get_bounding_box(self) -> NDArray[np.float64]:
        """
        Get axis-aligned bounding box containing the hypersphere.

        Returns:
            bounds: Array of shape (d, 2) where bounds[i] = [c_i - r, c_i + r]

        Example:
            >>> sphere = Hypersphere(center=[1, 2], radius=0.5)
            >>> bounds = sphere.get_bounding_box()
            >>> # array([[0.5, 1.5], [1.5, 2.5]])
        """
        bounds = np.zeros((self.dimension, 2))
        bounds[:, 0] = self.center - self.radius
        bounds[:, 1] = self.center + self.radius
        return bounds

    def sample_uniform(self, n_samples: int, max_attempts: int = 100, seed: int | None = None) -> NDArray:
        """
        Sample particles uniformly from hypersphere.

        Uses rejection sampling from bounding box. For high dimensions (d>10),
        most volume is near the surface, so acceptance rate drops.

        Args:
            n_samples: Number of particles
            max_attempts: Maximum attempts per particle
            seed: Random seed

        Returns:
            particles: Array of shape (n_samples, d)

        Complexity:
            Expected O(n_samples * d * (2/π)^(d/2)) attempts
            Acceptance rate ≈ V_sphere / V_box ≈ (π/4)^(d/2)

        Note:
            For d > 10, consider using surface sampling + radial distribution.

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> particles = sphere.sample_uniform(1000, seed=42)
            >>> # All particles satisfy ||p|| <= 1
            >>> assert np.all(np.linalg.norm(particles, axis=1) <= 1.0)
        """
        # Use parent class rejection sampling (efficient enough for d ≤ 10)
        return super().sample_uniform(n_samples, max_attempts, seed)

    def sample_surface(self, n_samples: int, seed: int | None = None) -> NDArray:
        """
        Sample particles uniformly on the surface of the hypersphere.

        Uses Muller method: Sample from Gaussian, normalize, scale by radius.

        Args:
            n_samples: Number of particles
            seed: Random seed

        Returns:
            surface_particles: Array of shape (n_samples, d) on boundary

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> surface = sphere.sample_surface(1000, seed=42)
            >>> # All particles have ||p|| = 1
            >>> dists = np.linalg.norm(surface, axis=1)
            >>> assert np.allclose(dists, 1.0)
        """
        if seed is not None:
            np.random.seed(seed)

        # Sample from standard Gaussian
        samples = np.random.normal(size=(n_samples, self.dimension))

        # Normalize to unit sphere
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        unit_sphere = samples / norms

        # Scale to radius and shift to center
        surface = self.center + self.radius * unit_sphere

        return surface

    def compute_volume(self, n_monte_carlo: int = 100000) -> float:
        """
        Compute exact volume (no Monte Carlo needed for hypersphere).

        Volume of d-dimensional ball with radius r:
            V_d(r) = (π^(d/2) / Γ(d/2 + 1)) * r^d

        Args:
            n_monte_carlo: Unused (kept for interface compatibility)

        Returns:
            Exact volume

        Example:
            >>> circle = Hypersphere(center=[0, 0], radius=1.0)
            >>> circle.compute_volume()  # π ≈ 3.14159...

            >>> sphere_3d = Hypersphere(center=[0, 0, 0], radius=1.0)
            >>> sphere_3d.compute_volume()  # 4π/3 ≈ 4.18879...
        """
        from math import gamma, pi

        # V_d(r) = (π^(d/2) / Γ(d/2 + 1)) * r^d
        coefficient = pi ** (self.dimension / 2) / gamma(self.dimension / 2 + 1)
        volume = coefficient * (self.radius**self.dimension)

        return volume

    def compute_surface_area(self) -> float:
        """
        Compute exact surface area of the hypersphere.

        Surface area of (d-1)-sphere with radius r:
            S_(d-1)(r) = d * V_d(r) / r

        Returns:
            Exact surface area

        Example:
            >>> circle = Hypersphere(center=[0, 0], radius=1.0)
            >>> circle.compute_surface_area()  # 2π (circumference)

            >>> sphere_3d = Hypersphere(center=[0, 0, 0], radius=1.0)
            >>> sphere_3d.compute_surface_area()  # 4π
        """
        volume = self.compute_volume()
        return self.dimension * volume / self.radius

    def __repr__(self) -> str:
        """String representation."""
        if self.dimension <= 3:
            center_str = str(self.center.tolist())
        else:
            center_str = f"[{self.center[0]:.2f}, {self.center[1]:.2f}, ..., {self.center[-1]:.2f}]"
        return f"Hypersphere(center={center_str}, radius={self.radius:.3f})"
