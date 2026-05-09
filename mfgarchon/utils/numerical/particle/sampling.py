"""
Monte Carlo utilities for MFG computations.

This module provides reusable Monte Carlo methods used across different
paradigms (numerical, neural, optimization, RL) for high-dimensional
integration, sampling, and variance reduction.

Key Components:
- Sampling strategies (uniform, stratified, importance, adaptive)
- Variance reduction techniques (control variates, antithetic variables)
- Low-discrepancy sequences (Sobol, Halton, Latin hypercube)
- High-dimensional integration methods
- Convergence diagnostics and error estimation

Mathematical Foundation:
- Standard MC: ∫f(x)dx ≈ (1/N) Σᵢ f(xᵢ), xᵢ ~ p(x)
- Importance MC: ∫f(x)dx = ∫(f(x)/p(x))p(x)dx
- Control Variates: Var[f - c(g - E[g])] ≤ Var[f]
- Stratified: Divide domain Ω = ∪Ωᵢ, sample within each stratum
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mfgarchon.utils.mfg_logging import get_logger

logger = get_logger(__name__)
# Type definitions for Monte Carlo integration
MCIntegrand = Callable[[NDArray], NDArray]
MCWeightFunction = Callable[[NDArray], NDArray]
MCDomain = list[tuple[float, float]]


@dataclass
class MCConfig:
    """Configuration for Monte Carlo computations."""

    # Sampling parameters
    num_samples: int = 10000
    sampling_method: str = "uniform"  # "uniform" | "stratified" | "importance" | "sobol" | "halton"
    seed: int | None = None

    # Variance reduction
    use_control_variates: bool = False
    use_antithetic_variables: bool = False
    control_function: Callable[[NDArray], NDArray] | None = None

    # Adaptive sampling
    adaptive: bool = False
    error_tolerance: float = 1e-3
    max_adaptive_samples: int = 100000

    # Stratification (for stratified sampling)
    num_strata_per_dim: int = 10

    # Importance sampling
    importance_function: Callable[[NDArray], NDArray] | None = None
    proposal_distribution: str = "uniform"  # "uniform" | "gaussian" | "custom"

    # Convergence monitoring
    batch_size: int = 1000
    confidence_level: float = 0.95

    # Computational
    parallel: bool = False
    chunk_size: int = 10000


@dataclass
class MCResult:
    """Result container for Monte Carlo computations."""

    # Estimation results
    estimate: float | NDArray
    standard_error: float | NDArray
    confidence_interval: tuple[float, float] | tuple[NDArray, NDArray]

    # Sampling information
    num_samples_used: int
    effective_sample_size: int
    variance_reduction_factor: float = 1.0

    # Convergence diagnostics
    converged: bool = False
    final_error: float = np.inf
    iterations: int = 0

    # Performance metrics
    computation_time: float = 0.0
    samples_per_second: float = 0.0


class MCSampler(ABC):
    """Abstract base class for Monte Carlo samplers."""

    def __init__(self, domain: MCDomain, config: MCConfig):
        """
        Initialize Monte Carlo sampler.

        Args:
            domain: List of (min, max) bounds for each dimension
            config: Monte Carlo configuration
        """
        self.domain = domain
        self.config = config
        self.dimension = len(domain)

        # Initialize random number generator
        self.rng = np.random.RandomState(config.seed)

        # Precompute domain properties
        self.domain_bounds = np.array(domain)
        self.domain_volume = np.prod(self.domain_bounds[:, 1] - self.domain_bounds[:, 0])

    @abstractmethod
    def sample(self, num_samples: int) -> NDArray:
        """
        Generate sample points in the domain.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Sample points of shape (num_samples, dimension)
        """

    def get_weights(self, points: NDArray) -> NDArray:
        """
        Get importance weights for sampled points.

        Args:
            points: Sample points

        Returns:
            Importance weights (default: uniform)
        """
        return np.ones(len(points)) / len(points)


class UniformMCSampler(MCSampler):
    """Uniform Monte Carlo sampler."""

    def sample(self, num_samples: int) -> NDArray:
        """Generate uniform random samples."""
        samples = np.zeros((num_samples, self.dimension))

        for i, (min_val, max_val) in enumerate(self.domain):
            samples[:, i] = self.rng.uniform(min_val, max_val, num_samples)

        return samples


class StratifiedMCSampler(MCSampler):
    """Stratified Monte Carlo sampler for variance reduction."""

    def sample(self, num_samples: int) -> NDArray:
        """Generate stratified samples."""
        strata_per_dim = self.config.num_strata_per_dim
        total_strata = strata_per_dim**self.dimension
        samples_per_stratum = max(1, num_samples // total_strata)

        samples = []

        # Generate stratified samples
        for stratum_idx in range(total_strata):
            # Convert linear index to multi-dimensional stratum coordinates
            stratum_coords = np.unravel_index(stratum_idx, (strata_per_dim,) * self.dimension)

            # Sample within this stratum
            stratum_samples = np.zeros((samples_per_stratum, self.dimension))

            for dim in range(self.dimension):
                min_val, max_val = self.domain[dim]
                stratum_width = (max_val - min_val) / strata_per_dim

                stratum_min = min_val + stratum_coords[dim] * stratum_width
                stratum_max = stratum_min + stratum_width

                stratum_samples[:, dim] = self.rng.uniform(stratum_min, stratum_max, samples_per_stratum)

            samples.append(stratum_samples)

        all_samples = np.vstack(samples)

        # Truncate to exact number if needed
        return all_samples[:num_samples]


class QuasiMCSampler(MCSampler):
    """Quasi-Monte Carlo sampler using low-discrepancy sequences."""

    def __init__(self, domain: MCDomain, config: MCConfig, sequence_type: str = "sobol"):
        """
        Initialize quasi-Monte Carlo sampler.

        Args:
            domain: Domain bounds
            config: MC configuration
            sequence_type: "sobol" | "halton" | "latin_hypercube"
        """
        super().__init__(domain, config)
        self.sequence_type = sequence_type
        self._init_sequence_generator()

    def _init_sequence_generator(self):
        """Initialize low-discrepancy sequence generator."""
        try:
            from scipy.stats import qmc

            if self.sequence_type == "sobol":
                self.qmc_sampler = qmc.Sobol(d=self.dimension, scramble=True, seed=self.config.seed)
            elif self.sequence_type == "halton":
                self.qmc_sampler = qmc.Halton(d=self.dimension, scramble=True, seed=self.config.seed)
            elif self.sequence_type == "latin_hypercube":
                self.qmc_sampler = qmc.LatinHypercube(d=self.dimension, seed=self.config.seed)
            else:
                raise ValueError(f"Unknown sequence type: {self.sequence_type}")

            self.scipy_available = True

        except ImportError:
            logger.warning("SciPy QMC not available, falling back to uniform sampling")
            self.scipy_available = False

    def sample(self, num_samples: int) -> NDArray:
        """Generate quasi-random samples."""
        if not self.scipy_available:
            # Fallback to uniform sampling
            return UniformMCSampler(self.domain, self.config).sample(num_samples)

        # Generate unit cube samples
        unit_samples = self.qmc_sampler.random(n=num_samples)

        # Transform to actual domain
        samples = np.zeros_like(unit_samples)
        for i, (min_val, max_val) in enumerate(self.domain):
            samples[:, i] = min_val + (max_val - min_val) * unit_samples[:, i]

        return samples


class ImportanceMCSampler(MCSampler):
    """Importance sampling Monte Carlo sampler."""

    def __init__(self, domain: MCDomain, config: MCConfig, importance_function: Callable[[NDArray], NDArray]):
        """
        Initialize importance sampling.

        Args:
            domain: Domain bounds
            config: MC configuration
            importance_function: Importance distribution p(x)
        """
        super().__init__(domain, config)
        self.importance_function = importance_function

    def sample(self, num_samples: int) -> NDArray:
        """Generate importance-weighted samples using rejection sampling."""
        # Use rejection sampling (simplified implementation)
        # In practice, more sophisticated methods would be used

        candidate_samples = num_samples * 3  # Generate extra candidates
        candidates = UniformMCSampler(self.domain, self.config).sample(candidate_samples)

        # Evaluate importance function
        importance_values = self.importance_function(candidates)
        max_importance = np.max(importance_values)

        # Rejection sampling
        accept_probs = importance_values / max_importance
        accepted_mask = self.rng.random(candidate_samples) < accept_probs

        accepted_samples = candidates[accepted_mask]

        # If not enough samples, fill with uniform
        if len(accepted_samples) < num_samples:
            additional_needed = num_samples - len(accepted_samples)
            additional_samples = UniformMCSampler(self.domain, self.config).sample(additional_needed)
            accepted_samples = np.vstack([accepted_samples, additional_samples])

        return accepted_samples[:num_samples]

    def get_weights(self, points: NDArray) -> NDArray:
        """Get importance weights (inverse of importance function)."""
        importance_vals = self.importance_function(points)
        weights = 1.0 / np.maximum(importance_vals, 1e-12)
        return weights / np.sum(weights)  # Normalize


class PoissonDiskSampler(MCSampler):
    """
    Poisson Disk Sampling using Bridson's algorithm.

    Guarantees minimum distance between points, preventing ill-conditioned
    GFDM stencils from point clustering. This is essential for meshfree
    methods where point spacing affects numerical stability.

    The algorithm uses a background grid for O(1) neighbor lookups,
    achieving O(N) overall complexity.

    Reference:
        Bridson, R. "Fast Poisson disk sampling in arbitrary dimensions."
        SIGGRAPH 2007.

    Example:
        >>> domain = [(0.0, 1.0), (0.0, 1.0)]
        >>> config = MCConfig(seed=42)
        >>> sampler = PoissonDiskSampler(domain, config, min_distance=0.05)
        >>> points = sampler.sample(500)
        >>> # All points have minimum separation >= 0.05
    """

    def __init__(
        self,
        domain: MCDomain,
        config: MCConfig,
        min_distance: float | None = None,
        k_candidates: int = 30,
    ):
        """
        Initialize Poisson Disk sampler.

        Args:
            domain: List of (min, max) bounds for each dimension
            config: Monte Carlo configuration
            min_distance: Minimum distance between points. If None, estimated
                         from domain size and target sample count.
            k_candidates: Number of candidate points to try around each active
                         point before rejection. Higher values give denser
                         packing but slower sampling. Default 30 is standard.
        """
        super().__init__(domain, config)
        self.min_distance = min_distance
        self.k_candidates = k_candidates

        # Precompute domain extents
        self._domain_min = np.array([b[0] for b in domain])
        self._domain_max = np.array([b[1] for b in domain])
        self._domain_size = self._domain_max - self._domain_min
        self._cached_offsets: list[tuple[int, ...]] | None = None

    def _estimate_min_distance(self, num_samples: int) -> float:
        """Estimate minimum distance from target sample count."""
        # For Poisson disk in d dimensions, approximate packing density
        # gives r ~ (V / N)^(1/d) where V is volume
        packing_factor = 0.7  # Conservative estimate
        return packing_factor * (self.domain_volume / num_samples) ** (1.0 / self.dimension)

    def _build_grid(self, r: float) -> tuple[NDArray, tuple[int, ...]]:
        """
        Build background grid for fast neighbor lookup.

        Args:
            r: Minimum distance

        Returns:
            Tuple of (grid array, grid shape)
        """
        # Cell size chosen so each cell can contain at most one point
        cell_size = r / np.sqrt(self.dimension)

        # Grid dimensions
        grid_shape = tuple(max(1, int(np.ceil(s / cell_size))) for s in self._domain_size)

        # Initialize grid with -1 (empty)
        grid = np.full(grid_shape, -1, dtype=np.int32)

        return grid, grid_shape, cell_size

    def _point_to_grid_coords(self, point: NDArray, cell_size: float) -> tuple[int, ...]:
        """Convert point coordinates to grid cell indices."""
        coords = ((point - self._domain_min) / cell_size).astype(int)
        # Clamp to valid range
        coords = np.clip(coords, 0, np.array(self._grid_shape) - 1)
        return tuple(coords)

    def _is_valid_point(
        self,
        point: NDArray,
        points: list[NDArray],
        grid: NDArray,
        cell_size: float,
        r: float,
    ) -> bool:
        """
        Check if a candidate point is valid (far enough from all neighbors).

        Uses background grid for O(1) average case neighbor lookup.
        """
        # Check domain bounds
        if np.any(point < self._domain_min) or np.any(point >= self._domain_max):
            return False

        # Get grid cell
        cell_coords = self._point_to_grid_coords(point, cell_size)

        # Check neighboring cells (within 2 cells in each direction)
        r_squared = r * r

        # Generate neighbor cell offsets
        for offset in self._get_neighbor_offsets():
            neighbor_coords = tuple(c + o for c, o in zip(cell_coords, offset, strict=True))

            # Check bounds
            if any(c < 0 or c >= s for c, s in zip(neighbor_coords, self._grid_shape, strict=True)):
                continue

            # Check if cell contains a point
            point_idx = grid[neighbor_coords]
            if point_idx >= 0:
                # Check distance
                existing_point = points[point_idx]
                dist_squared = np.sum((point - existing_point) ** 2)
                if dist_squared < r_squared:
                    return False

        return True

    def _get_neighbor_offsets(self) -> list[tuple[int, ...]]:
        """Generate all neighbor cell offsets within distance 2."""
        if self._cached_offsets is None:
            # Generate all combinations of -2, -1, 0, 1, 2 for each dimension
            from itertools import product

            self._cached_offsets = list(product(range(-2, 3), repeat=self.dimension))
        return self._cached_offsets

    def sample(self, num_samples: int) -> NDArray:
        """
        Generate well-spaced points using Bridson's algorithm.

        Args:
            num_samples: Target number of samples. Actual count may be less
                        if the domain cannot fit that many points with the
                        given minimum distance.

        Returns:
            Array of shape (n_actual, dimension) with well-spaced points
        """
        # Determine minimum distance
        r = self.min_distance
        if r is None:
            r = self._estimate_min_distance(num_samples)

        # Build background grid
        grid, self._grid_shape, cell_size = self._build_grid(r)

        # Initialize with random first point
        points: list[NDArray] = []
        active_list: list[int] = []

        first_point = self._domain_min + self.rng.random(self.dimension) * self._domain_size
        points.append(first_point)
        active_list.append(0)

        # Mark grid cell
        cell = self._point_to_grid_coords(first_point, cell_size)
        grid[cell] = 0

        # Main loop: process active points
        while active_list and len(points) < num_samples:
            # Pick random active point
            active_idx = self.rng.randint(len(active_list))
            point_idx = active_list[active_idx]
            center = points[point_idx]

            # Try k candidates around this point
            found = False
            for _ in range(self.k_candidates):
                # Generate random point in annulus [r, 2r]
                candidate = self._generate_annulus_point(center, r)

                if self._is_valid_point(candidate, points, grid, cell_size, r):
                    # Accept candidate
                    new_idx = len(points)
                    points.append(candidate)
                    active_list.append(new_idx)

                    # Mark grid
                    cell = self._point_to_grid_coords(candidate, cell_size)
                    grid[cell] = new_idx

                    found = True
                    break

            if not found:
                # Remove from active list
                active_list.pop(active_idx)

        result = np.array(points)
        logger.debug(f"PoissonDisk: generated {len(result)} points (target: {num_samples})")

        return result

    def _generate_annulus_point(self, center: NDArray, r: float) -> NDArray:
        """Generate random point in annulus [r, 2r] around center."""
        # Generate random direction (uniform on unit sphere)
        direction = self.rng.randn(self.dimension)
        direction /= np.linalg.norm(direction)

        # Generate random radius in [r, 2r]
        # Use r * (1 + random) for uniform distribution in annulus
        radius = r * (1.0 + self.rng.random())

        return center + radius * direction


class ControlVariates:
    """Control variates for variance reduction."""

    def __init__(self, control_function: Callable[[NDArray], NDArray]):
        """
        Initialize control variates.

        Args:
            control_function: Known function g(x) with known expectation
        """
        self.control_function = control_function
        self.control_coefficient = 0.0
        self.control_mean = 0.0
        self.is_calibrated = False

    def calibrate(self, sample_points: NDArray, target_values: NDArray, control_expectation: float | None = None):
        """
        Calibrate control variate coefficient.

        Args:
            sample_points: Sample points for calibration
            target_values: Target function values f(x)
            control_expectation: Known E[g] (estimated if None)
        """
        control_values = self.control_function(sample_points)

        if control_expectation is None:
            self.control_mean = np.mean(control_values)
        else:
            self.control_mean = control_expectation

        # Compute optimal coefficient: c* = Cov[f,g] / Var[g]
        covariance = np.cov(target_values, control_values)[0, 1]
        control_variance = np.var(control_values)

        if control_variance > 1e-12:
            self.control_coefficient = covariance / control_variance
            self.is_calibrated = True

            # Compute variance reduction factor
            correlation = covariance / (np.std(target_values) * np.std(control_values))
            variance_reduction = max(0, 1 - correlation**2)

            logger.info(f"Control variate calibrated: c* = {self.control_coefficient:.6f}")
            logger.info(f"Expected variance reduction: {variance_reduction:.3f}")
        else:
            logger.warning("Control function has zero variance")

    def apply(self, sample_points: NDArray, target_values: NDArray) -> NDArray:
        """Apply control variate variance reduction."""
        if not self.is_calibrated:
            return target_values

        control_values = self.control_function(sample_points)
        reduced_values = target_values - self.control_coefficient * (control_values - self.control_mean)

        return reduced_values


def monte_carlo_integrate(integrand: MCIntegrand, domain: MCDomain, config: MCConfig | None = None) -> MCResult:
    """
    Monte Carlo integration of a function over a domain.

    Args:
        integrand: Function f(x) to integrate
        domain: Integration domain as list of (min, max) bounds
        config: Monte Carlo configuration

    Returns:
        MCResult with integration estimate and diagnostics
    """
    if config is None:
        config = MCConfig()

    import time

    start_time = time.time()

    # Choose sampler based on configuration
    sampler: MCSampler
    if config.sampling_method == "uniform":
        sampler = UniformMCSampler(domain, config)
    elif config.sampling_method == "stratified":
        sampler = StratifiedMCSampler(domain, config)
    elif config.sampling_method in ["sobol", "halton", "latin_hypercube"]:
        sampler = QuasiMCSampler(domain, config, config.sampling_method)
    else:
        logger.warning(f"Unknown sampling method: {config.sampling_method}, using uniform")
        sampler = UniformMCSampler(domain, config)

    # Sample points
    sample_points = sampler.sample(config.num_samples)
    weights = sampler.get_weights(sample_points)

    # Evaluate integrand
    function_values = integrand(sample_points)

    # Apply variance reduction if configured
    if config.use_control_variates and config.control_function is not None:
        cv = ControlVariates(config.control_function)
        cv.calibrate(sample_points, function_values)
        function_values = cv.apply(sample_points, function_values)

    # Apply antithetic variables if configured
    if config.use_antithetic_variables:
        function_values = _apply_antithetic_variables(function_values, sampler, integrand)

    # Compute estimate
    domain_volume = np.prod([max_val - min_val for min_val, max_val in domain])
    weighted_values = function_values * weights * len(weights)
    estimate = domain_volume * np.mean(weighted_values)

    # Compute error estimates
    standard_error = domain_volume * np.std(weighted_values) / np.sqrt(len(weighted_values))

    # Confidence interval
    z_score = 1.96 if config.confidence_level == 0.95 else 2.576  # Simplified
    ci_lower = estimate - z_score * standard_error
    ci_upper = estimate + z_score * standard_error

    # Create result
    result = MCResult(
        estimate=estimate,
        standard_error=standard_error,
        confidence_interval=(ci_lower, ci_upper),
        num_samples_used=config.num_samples,
        effective_sample_size=config.num_samples,
        converged=standard_error < config.error_tolerance,
        final_error=standard_error,
        computation_time=time.time() - start_time,
        samples_per_second=config.num_samples / max(time.time() - start_time, 1e-6),
    )

    return result


def _apply_antithetic_variables(function_values: NDArray, sampler: MCSampler, integrand: MCIntegrand) -> NDArray:
    """Apply antithetic variables variance reduction."""
    # Generate antithetic samples (simplified)
    # For uniform distribution: if U ~ Uniform(0,1), then 1-U is antithetic
    # This is a simplified implementation
    return function_values  # Placeholder


def adaptive_monte_carlo(
    integrand: MCIntegrand,
    domain: MCDomain,
    target_error: float = 1e-3,
    initial_samples: int = 1000,
    max_samples: int = 100000,
    batch_size: int = 1000,
) -> MCResult:
    """
    Adaptive Monte Carlo integration with error control.

    Args:
        integrand: Function to integrate
        domain: Integration domain
        target_error: Target error tolerance
        initial_samples: Initial number of samples
        max_samples: Maximum total samples
        batch_size: Batch size for adaptive sampling

    Returns:
        MCResult with adaptive sampling statistics
    """
    config = MCConfig(num_samples=initial_samples)

    # Initial estimation
    result = monte_carlo_integrate(integrand, domain, config)

    total_samples = initial_samples
    estimates = [result.estimate]
    errors = [result.standard_error]

    # Adaptive refinement
    while result.standard_error > target_error and total_samples < max_samples:
        # Add more samples
        additional_samples = min(batch_size, max_samples - total_samples)
        config.num_samples = additional_samples

        batch_result = monte_carlo_integrate(integrand, domain, config)

        # Combine estimates (simplified combination)
        total_samples += additional_samples
        weight_old = len(estimates) * initial_samples
        weight_new = additional_samples

        combined_estimate = (weight_old * result.estimate + weight_new * batch_result.estimate) / (
            weight_old + weight_new
        )

        # Update error estimate (simplified)
        result.estimate = combined_estimate
        result.standard_error *= np.sqrt(len(estimates) * initial_samples / total_samples)
        result.num_samples_used = total_samples

        estimates.append(combined_estimate)
        errors.append(result.standard_error)

        logger.info(f"Adaptive MC: {total_samples} samples, error = {result.standard_error:.6e}")

    result.converged = result.standard_error <= target_error
    result.iterations = len(estimates)

    return result


# Convenience functions for common use cases
def integrate_gaussian_quadrature_mc(
    mean: NDArray, cov: NDArray, integrand: MCIntegrand, num_samples: int = 10000
) -> MCResult:
    """Monte Carlo integration with Gaussian distribution."""
    # Create domain (6-sigma bounds) - vectorized
    sigma = np.sqrt(np.diag(cov))
    lower_bounds = mean - 6 * sigma
    upper_bounds = mean + 6 * sigma
    domain = list(zip(lower_bounds.tolist(), upper_bounds.tolist(), strict=True))

    # Gaussian importance function
    # Issue #1066: use solve() not inv() — for high-dim cov with cond > 1e10,
    # inv()-based Mahalanobis silently returns wrong values. solve() is stable.
    def gaussian_importance(x):
        diff = x - mean
        # cov is symmetric → diff @ cov^{-1} = solve(cov, diff.T).T
        sol = np.linalg.solve(cov, diff.T).T
        return np.exp(-0.5 * np.einsum("ij,ij->i", sol, diff))

    config = MCConfig(num_samples=num_samples, sampling_method="importance", importance_function=gaussian_importance)

    return monte_carlo_integrate(integrand, domain, config)


def estimate_expectation(samples: NDArray, weights: NDArray | None = None) -> tuple[float, float]:
    """
    Estimate expectation and standard error from samples.

    Args:
        samples: Sample values
        weights: Sample weights (uniform if None)

    Returns:
        Tuple of (expectation, standard_error)
    """
    if weights is None:
        weights = np.ones(len(samples)) / len(samples)

    expectation = np.average(samples, weights=weights)
    variance = np.average((samples - expectation) ** 2, weights=weights)
    standard_error = np.sqrt(variance / len(samples))

    return expectation, standard_error


def sample_from_scattered_density(
    points: NDArray,
    density: NDArray,
    num_samples: int,
    seed: int | None = None,
    periodic_bounds: list[tuple[float, float]] | None = None,
    periodic_dims: tuple[int, ...] | None = None,
) -> NDArray:
    """
    Sample particles from density defined at scattered (meshfree) points.

    Uses Delaunay triangulation to partition the convex hull of the points,
    then samples simplices proportional to their integrated density mass,
    and finally samples uniformly within each selected simplex.

    This is the meshfree analog of `sample_from_density` for use with
    GFDM collocation points or other irregular point distributions.

    Parameters
    ----------
    points : ndarray
        Scattered point coordinates, shape (N_points, dimension).
        Points should cover the sampling region.
    density : ndarray
        Density values at each point, shape (N_points,).
        Must be non-negative. Will be normalized internally.
    num_samples : int
        Number of particle samples to generate.
    seed : int or None, default=None
        Random seed for reproducibility.
    periodic_bounds : list of (float, float), optional
        Domain bounds for periodic topology, e.g. [(0, 1), (0, 1)].
        If provided, uses ghost point augmentation to create valid
        Delaunay triangles across periodic boundaries (Issue #714).
    periodic_dims : tuple of int, optional
        Dimensions with periodic topology. If None and periodic_bounds
        is provided, all dimensions are treated as periodic.

    Returns
    -------
    particles : ndarray
        Sampled particle positions, shape (num_samples, dimension).
        For periodic domains, positions are wrapped to the fundamental domain.

    Examples
    --------
    Sample from Gaussian density at scattered collocation points:

    >>> import numpy as np
    >>> # Create scattered points via Poisson disk sampling
    >>> rng = np.random.default_rng(42)
    >>> points = rng.random((200, 2))  # 200 scattered points in [0,1]²
    >>> # Gaussian density centered at (0.5, 0.5)
    >>> density = np.exp(-10 * ((points[:, 0] - 0.5)**2 + (points[:, 1] - 0.5)**2))
    >>> particles = sample_from_scattered_density(points, density, num_samples=1000)
    >>> # particles will cluster around (0.5, 0.5)

    Periodic sampling on torus:

    >>> # Density peak near boundary (wraps around on torus)
    >>> density_periodic = np.exp(-10 * (points[:, 0] - 0.05)**2)
    >>> particles = sample_from_scattered_density(
    ...     points, density_periodic, num_samples=500,
    ...     periodic_bounds=[(0, 1), (0, 1)], periodic_dims=(0, 1)
    ... )

    Notes
    -----
    - Without periodic: only samples within convex hull of input points
    - With periodic: samples across boundary using ghost point augmentation,
      then wraps results to fundamental domain (Issue #714)
    - If all densities are zero, falls back to uniform sampling over simplices
    - Computational complexity: O(N_points log N_points) for triangulation,
      O(num_samples) for sampling
    """
    import math

    from scipy.spatial import Delaunay

    rng = np.random.RandomState(seed)
    N_points, dimension = points.shape

    if len(density) != N_points:
        raise ValueError(f"Density length {len(density)} != number of points {N_points}")

    # Ensure non-negative
    density = np.maximum(density, 0.0)

    # Issue #714: Handle periodic topology via ghost point augmentation
    if periodic_bounds is not None:
        from mfgarchon.geometry.boundary.periodic import create_periodic_ghost_points, wrap_positions

        # Default: all dimensions periodic if bounds provided but dims not specified
        if periodic_dims is None:
            periodic_dims = tuple(range(dimension))

        # Create augmented point cloud with ghost copies
        aug_points, orig_indices = create_periodic_ghost_points(points, periodic_bounds, periodic_dims)

        # Extend density to ghost points (they have same density as originals)
        density_aug = density[orig_indices]

        # Build Delaunay on augmented points
        tri = Delaunay(aug_points)
        simplices = tri.simplices

        # Filter simplices: only keep those with centroid in original domain
        # This avoids sampling from "redundant" triangles in ghost regions
        bounds_arr = np.array(periodic_bounds)
        valid_simplices = []
        valid_simplex_masses = []

        for simplex in simplices:
            vertices = aug_points[simplex]
            centroid = np.mean(vertices, axis=0)

            # Check if centroid is in original domain
            in_domain = True
            for d in range(dimension):
                if centroid[d] < bounds_arr[d, 0] or centroid[d] > bounds_arr[d, 1]:
                    in_domain = False
                    break

            if in_domain:
                # Compute mass for this simplex
                avg_density = np.mean(density_aug[simplex])
                edge_matrix = vertices[1:] - vertices[0]
                volume = np.abs(np.linalg.det(edge_matrix)) / math.factorial(dimension)
                valid_simplices.append(simplex)
                valid_simplex_masses.append(avg_density * volume)

        simplices = np.array(valid_simplices)
        simplex_masses = np.array(valid_simplex_masses)
        N_simplices = len(simplices)
        points_for_sampling = aug_points
        logger.debug(
            f"sample_from_scattered_density: periodic mode, "
            f"{N_points} -> {len(aug_points)} augmented points, "
            f"{N_simplices} valid simplices"
        )
    else:
        # Standard non-periodic case
        tri = Delaunay(points)
        simplices = tri.simplices
        N_simplices = len(simplices)
        points_for_sampling = points

        # Compute mass for each simplex
        simplex_masses = np.zeros(N_simplices)
        for i, simplex in enumerate(simplices):
            vertices = points[simplex]
            avg_density = np.mean(density[simplex])
            edge_matrix = vertices[1:] - vertices[0]
            volume = np.abs(np.linalg.det(edge_matrix)) / math.factorial(dimension)
            simplex_masses[i] = avg_density * volume

    total_mass = np.sum(simplex_masses)

    if total_mass < 1e-14:
        # Uniform fallback: equal probability for each simplex
        simplex_probs = np.ones(N_simplices) / N_simplices
        logger.debug("sample_from_scattered_density: zero total mass, using uniform fallback")
    else:
        simplex_probs = simplex_masses / total_mass

    # Sample simplex indices according to mass distribution
    simplex_indices = rng.choice(N_simplices, size=num_samples, p=simplex_probs, replace=True)

    # Sample uniformly within each selected simplex
    # Use barycentric coordinates with Dirichlet distribution
    particles = np.zeros((num_samples, dimension))

    for i, simplex_idx in enumerate(simplex_indices):
        vertices = points_for_sampling[simplices[simplex_idx]]  # Shape: (d+1, d)

        # Uniform sampling in simplex via sorted uniform variates
        # Generate d uniform variates, sort, take differences -> barycentric coords
        u = np.sort(rng.random(dimension))
        barycentric = np.concatenate([[u[0]], np.diff(u), [1.0 - u[-1]]])

        # Convert to Cartesian
        particles[i] = barycentric @ vertices

    # Issue #714: Wrap particles to fundamental domain if periodic
    if periodic_bounds is not None:
        particles = wrap_positions(particles, periodic_bounds, periodic_dims)

    return particles


def sample_from_density(
    density: NDArray,
    coordinates: list[NDArray],
    num_samples: int,
    jitter: bool = True,
    seed: int | None = None,
) -> NDArray:
    """
    Sample particle positions from a density field on a regular grid.

    Uses inverse CDF sampling: treats the density as a probability distribution
    over grid cells, samples cell indices, then optionally adds sub-grid jitter
    for smoothness.

    This is the complement of KDE: given particles, KDE estimates density on grid;
    given density on grid, this function samples particles.

    Parameters
    ----------
    density : ndarray
        Density values on regular grid, shape (N1, N2, ..., Nd).
        Must be non-negative. Will be normalized internally.
    coordinates : list of ndarray
        1D coordinate arrays for each dimension, e.g. [x_coords, y_coords].
        Each array has length matching the corresponding density dimension.
    num_samples : int
        Number of particle samples to generate.
    jitter : bool, default=True
        If True, add uniform sub-grid jitter for smoothness.
        If False, return exact grid cell centers.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    particles : ndarray
        Sampled particle positions, shape (num_samples, dimension).

    Examples
    --------
    1D sampling from Gaussian density:

    >>> x = np.linspace(0, 1, 51)
    >>> density = np.exp(-20 * (x - 0.5)**2)
    >>> particles = sample_from_density(density, [x], num_samples=1000)
    >>> # particles will cluster around x=0.5

    2D sampling:

    >>> x = np.linspace(0, 1, 31)
    >>> y = np.linspace(0, 1, 31)
    >>> X, Y = np.meshgrid(x, y, indexing='ij')
    >>> density = np.exp(-10 * ((X - 0.5)**2 + (Y - 0.5)**2))
    >>> particles = sample_from_density(density, [x, y], num_samples=500)
    >>> # particles will cluster around (0.5, 0.5)

    Notes
    -----
    - If density is all zeros, falls back to uniform sampling over the domain.
    - Sub-grid jitter prevents aliasing artifacts in visualization and improves
      the smoothness of subsequent KDE density estimation.
    """
    rng = np.random.RandomState(seed)
    dimension = len(coordinates)
    grid_shape = tuple(len(c) for c in coordinates)

    # Validate density shape
    if density.shape != grid_shape:
        raise ValueError(f"Density shape {density.shape} does not match coordinates shape {grid_shape}")

    # Flatten density and normalize to probability
    density_flat = density.ravel()
    total_mass = np.sum(density_flat)

    if total_mass < 1e-14:
        # Uniform fallback if density is zero
        bounds = np.array([[c[0], c[-1]] for c in coordinates])  # shape: (d, 2)
        particles = rng.uniform(bounds[:, 0], bounds[:, 1], size=(num_samples, dimension))
        return particles

    probs = density_flat / total_mass

    # Sample flat indices according to probability
    try:
        flat_indices = rng.choice(len(density_flat), size=num_samples, p=probs, replace=True)
    except ValueError:
        # Fallback to uniform if probability is degenerate
        bounds = np.array([[c[0], c[-1]] for c in coordinates])
        particles = rng.uniform(bounds[:, 0], bounds[:, 1], size=(num_samples, dimension))
        return particles

    # Convert flat indices to multi-indices
    multi_indices = np.unravel_index(flat_indices, grid_shape)

    # Get coordinates at sampled grid points (vectorized)
    particles = np.column_stack([coordinates[d][multi_indices[d]] for d in range(dimension)])

    # Add sub-grid jitter for smoothness
    if jitter:
        spacings = np.array([c[1] - c[0] if len(c) > 1 else 0.0 for c in coordinates])
        jitter_offsets = rng.uniform(-0.5, 0.5, size=(num_samples, dimension)) * spacings
        particles += jitter_offsets

    return particles


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for sampling utilities."""
    print("Testing sampling utilities...")

    # Test 1: sample_from_density 1D
    print("\n1. Testing sample_from_density (1D)...")
    x = np.linspace(0, 1, 51)
    density_1d = np.exp(-20 * (x - 0.5) ** 2)  # Gaussian at 0.5

    particles_1d = sample_from_density(density_1d, [x], num_samples=1000, seed=42)
    assert particles_1d.shape == (1000, 1), f"Expected (1000, 1), got {particles_1d.shape}"
    mean_1d = np.mean(particles_1d)
    assert 0.4 < mean_1d < 0.6, f"Mean should be near 0.5, got {mean_1d:.3f}"
    print(f"  1D: mean = {mean_1d:.3f} (expected ~0.5)")

    # Test 2: sample_from_density 2D
    print("\n2. Testing sample_from_density (2D)...")
    x2 = np.linspace(0, 1, 31)
    y2 = np.linspace(0, 1, 31)
    X, Y = np.meshgrid(x2, y2, indexing="ij")
    density_2d = np.exp(-10 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

    particles_2d = sample_from_density(density_2d, [x2, y2], num_samples=500, seed=42)
    assert particles_2d.shape == (500, 2), f"Expected (500, 2), got {particles_2d.shape}"
    mean_x = np.mean(particles_2d[:, 0])
    mean_y = np.mean(particles_2d[:, 1])
    assert 0.4 < mean_x < 0.6, f"Mean x should be near 0.5, got {mean_x:.3f}"
    assert 0.4 < mean_y < 0.6, f"Mean y should be near 0.5, got {mean_y:.3f}"
    print(f"  2D: mean = ({mean_x:.3f}, {mean_y:.3f}) (expected ~(0.5, 0.5))")

    # Test 3: sample_from_density with zero density (uniform fallback)
    print("\n3. Testing zero density fallback...")
    zero_density = np.zeros((21,))
    particles_zero = sample_from_density(zero_density, [np.linspace(0, 1, 21)], num_samples=100, seed=42)
    assert particles_zero.shape == (100, 1)
    # Should be uniformly distributed
    mean_zero = np.mean(particles_zero)
    assert 0.3 < mean_zero < 0.7, f"Uniform mean should be near 0.5, got {mean_zero:.3f}"
    print(f"  Zero density fallback: mean = {mean_zero:.3f}")

    # Test 4: sample_from_density without jitter
    print("\n4. Testing no-jitter mode...")
    particles_no_jitter = sample_from_density(density_1d, [x], num_samples=100, jitter=False, seed=42)
    # Without jitter, particles should be exactly on grid points
    spacings = x[1] - x[0]
    # Check that all particles are within grid bounds
    assert np.all(particles_no_jitter >= x[0]), "Particles should be >= domain min"
    assert np.all(particles_no_jitter <= x[-1]), "Particles should be <= domain max"
    print("  No-jitter mode works correctly")

    # Test 5: PoissonDiskSampler
    print("\n5. Testing PoissonDiskSampler...")
    domain = [(0.0, 1.0), (0.0, 1.0)]
    config = MCConfig(seed=42)
    sampler = PoissonDiskSampler(domain, config, min_distance=0.05)
    poisson_points = sampler.sample(200)
    assert poisson_points.shape[1] == 2, "Should be 2D points"
    assert len(poisson_points) > 50, "Should generate reasonable number of points"
    print(f"  PoissonDiskSampler: generated {len(poisson_points)} points")

    # Verify minimum distance constraint
    from scipy.spatial import distance_matrix

    dist_matrix = distance_matrix(poisson_points, poisson_points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist = np.min(dist_matrix)
    assert min_dist >= 0.04, f"Minimum distance should be ~0.05, got {min_dist:.3f}"
    print(f"  Minimum point separation: {min_dist:.3f}")

    # Test 6: sample_from_scattered_density (meshfree)
    print("\n6. Testing sample_from_scattered_density (meshfree)...")
    # Create scattered points
    scattered_points = np.random.rand(200, 2)  # 200 random points in [0,1]²
    # Gaussian density centered at (0.5, 0.5)
    scattered_density = np.exp(-10 * ((scattered_points[:, 0] - 0.5) ** 2 + (scattered_points[:, 1] - 0.5) ** 2))

    particles_scattered = sample_from_scattered_density(scattered_points, scattered_density, num_samples=500, seed=42)
    assert particles_scattered.shape == (500, 2), f"Expected (500, 2), got {particles_scattered.shape}"
    mean_x_scat = np.mean(particles_scattered[:, 0])
    mean_y_scat = np.mean(particles_scattered[:, 1])
    assert 0.35 < mean_x_scat < 0.65, f"Mean x should be near 0.5, got {mean_x_scat:.3f}"
    assert 0.35 < mean_y_scat < 0.65, f"Mean y should be near 0.5, got {mean_y_scat:.3f}"
    print(f"  Scattered 2D: mean = ({mean_x_scat:.3f}, {mean_y_scat:.3f}) (expected ~(0.5, 0.5))")

    # Test 7: sample_from_scattered_density with uniform density
    print("\n7. Testing scattered sampling with uniform density...")
    uniform_density = np.ones(len(scattered_points))
    particles_uniform = sample_from_scattered_density(scattered_points, uniform_density, num_samples=500, seed=42)
    mean_uniform = np.mean(particles_uniform, axis=0)
    # Should be roughly centered (depends on point distribution)
    assert 0.3 < mean_uniform[0] < 0.7, f"Mean x should be near 0.5, got {mean_uniform[0]:.3f}"
    print(f"  Uniform density: mean = ({mean_uniform[0]:.3f}, {mean_uniform[1]:.3f})")

    # Test 8: Periodic sampling (Issue #714)
    print("\n8. Testing periodic sampling (Issue #714)...")
    # Create points covering [0,1]²
    np.random.seed(789)
    periodic_points = np.random.rand(150, 2)
    # Density peak near left boundary (x ~ 0.05)
    # On a torus, this should also attract samples from x ~ 0.95 wrapping around
    periodic_density = np.exp(-50 * (periodic_points[:, 0] - 0.05) ** 2)

    # Sample without periodic handling
    particles_non_periodic = sample_from_scattered_density(periodic_points, periodic_density, num_samples=500, seed=42)

    # Sample with periodic handling
    periodic_bounds = [(0.0, 1.0), (0.0, 1.0)]
    particles_periodic = sample_from_scattered_density(
        periodic_points,
        periodic_density,
        num_samples=500,
        seed=42,
        periodic_bounds=periodic_bounds,
        periodic_dims=(0, 1),
    )

    # Both should cluster near x=0.05
    mean_non_periodic = np.mean(particles_non_periodic[:, 0])
    mean_periodic = np.mean(particles_periodic[:, 0])
    print(f"  Non-periodic mean x: {mean_non_periodic:.3f}")
    print(f"  Periodic mean x: {mean_periodic:.3f}")

    # Verify all periodic samples are within bounds
    assert np.all(particles_periodic[:, 0] >= 0.0), "Periodic samples should be >= 0"
    assert np.all(particles_periodic[:, 0] <= 1.0), "Periodic samples should be <= 1"
    assert np.all(particles_periodic[:, 1] >= 0.0), "Periodic samples should be >= 0"
    assert np.all(particles_periodic[:, 1] <= 1.0), "Periodic samples should be <= 1"
    print("  All periodic samples within bounds: passed")

    # Test cylinder topology (periodic only in x)
    particles_cylinder = sample_from_scattered_density(
        periodic_points,
        periodic_density,
        num_samples=500,
        seed=42,
        periodic_bounds=periodic_bounds,
        periodic_dims=(0,),
    )
    assert np.all(particles_cylinder[:, 0] >= 0.0), "Cylinder samples should be >= 0"
    assert np.all(particles_cylinder[:, 0] <= 1.0), "Cylinder samples should be <= 1"
    print("  Cylinder topology: passed")
    print("  Periodic sampling (Issue #714): OK")

    print("\nAll smoke tests passed!")
