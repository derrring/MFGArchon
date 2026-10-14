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

from mfgarchon.utils.deprecation import NEVER_READ, retired_parameters
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
    sampling_method: str = "uniform"  # one of SAMPLING_METHODS; any other is refused
    seed: int | None = None

    # Variance reduction
    use_control_variates: bool = False
    use_antithetic_variables: bool = False
    control_function: Callable[[NDArray], NDArray] | None = None

    # Adaptive sampling
    adaptive: bool = NEVER_READ  # refused when passed; never read (#2648)
    error_tolerance: float = 1e-3
    max_adaptive_samples: int = NEVER_READ  # refused when passed; never read (#2648)

    # Stratification (for stratified sampling)
    num_strata_per_dim: int = 10

    # Importance sampling
    importance_function: Callable[[NDArray], NDArray] | None = None
    proposal_distribution: str = NEVER_READ  # refused when passed; never read (#2648)

    # Convergence monitoring
    batch_size: int = NEVER_READ  # refused when passed; never read (#2648)
    confidence_level: float = 0.95

    # Computational
    parallel: bool = NEVER_READ  # refused when passed; never read (#2648)
    chunk_size: int = NEVER_READ  # refused when passed; never read (#2648)


MCConfig.__init__ = retired_parameters(  # type: ignore[method-assign]
    {
        "adaptive": "monte_carlo_integrate samples once; call adaptive_monte_carlo for adaptive sampling",
        "max_adaptive_samples": "adaptive_monte_carlo takes max_samples= itself",
        "proposal_distribution": "no sampler reads a proposal distribution",
        "batch_size": "adaptive_monte_carlo takes batch_size= itself",
        "chunk_size": "monte_carlo_integrate evaluates the integrand on all samples at once",
        "parallel": "the integrand is evaluated in this process",
    },
    issue="#2648",
    since="v0.22.0",
    never_read=True,
)(MCConfig.__init__)


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
        """Generate stratified samples: ``num_samples // strata**d`` uniform points in every stratum.

        Every stratum must get a point, so ``num_strata_per_dim**d`` greater than ``num_samples`` is refused (#2648):
        it used to keep the first ``num_samples`` strata in C order, which confines the first coordinate to its low
        strata -- at d = 5 with the defaults, the mean of x0 came back 0.05 against 0.5. When ``num_samples`` is not
        a multiple of the strata, the remainder is not drawn, and the estimate uses the points that were.
        """
        strata_per_dim = self.config.num_strata_per_dim
        total_strata = strata_per_dim**self.dimension
        if total_strata > num_samples:
            raise ValueError(
                f"StratifiedMCSampler: {strata_per_dim} strata per dimension in d = {self.dimension} make "
                f"{total_strata} strata, more than the {num_samples} samples, so some strata would get no point and "
                f"the estimate would cover only part of the domain. Use num_strata_per_dim <= "
                f"{int(np.floor(num_samples ** (1.0 / self.dimension) + 1e-9))}, or num_samples >= {total_strata}."
            )
        samples_per_stratum = num_samples // total_strata

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

        return np.vstack(samples)


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
    """Importance sampling: draw from a density p on the domain and estimate an integral as mean(f / p).

    Contract (#2648, ruling 8): ``importance_function`` is a probability density on the domain D -- finite,
    non-negative, positive wherever the integrand is non-zero -- whose integral Z over D is 1. The expectation of the
    estimate is then the integral of f over D divided by Z, which is unbiased at Z = 1, and a density matched to the
    integrand gives a zero-variance estimate. A density that vanishes where f does not misses that part of the
    integral; the sampler keeps its check points where p = 0, and ``monte_carlo_integrate`` estimates the missing part
    there (`_refuse_integrand_outside_the_support`).

    - **``NORMALIZATION_TOL``** is the relative error in the integral's mass that the estimator may carry silently. It
      is read twice: as the floor of the normalization window, and as the largest fraction of the integral of |f|
      that may lie where p = 0.
    - **Normalization** is checked at construction, by a fixed-seed Monte Carlo estimate of Z over
      ``_CHECK_SAMPLES`` points. It is refused when Z differs from 1 by more than both ``NORMALIZATION_TOL`` and 4 of
      the check's standard errors. A check of finite size cannot certify Z more closely than its own error, so the
      window the estimate may still carry is max(``NORMALIZATION_TOL``, 4 standard errors), relative. The check
      draws uniformly, so for a sharply peaked density its error, and the window with it, is large.
    - **Non-finite or negative values** at the check points are refused.
    - **Rejection sampling** draws exactly from p: uniform candidates are accepted with probability p(x) / M.
      M is ``importance_max`` when given; otherwise it is the largest p over a dense draw, times ``ENVELOPE_SAFETY``.
      Every candidate is checked, and one with p(x) > M is refused, naming the bound and the point, because
      rejection with too small an M silently samples min(p, M) instead of p.
    - **An infeasible acceptance rate** is refused up front. It is 1 / (V M) for a normalized p on a domain of
      volume V.
    """

    NORMALIZATION_TOL = 1e-6
    ENVELOPE_SAFETY = 1.5
    MIN_ACCEPTANCE = 1e-4
    _CHECK_SAMPLES = 100_000
    _CHECK_SEED = 2648

    def __init__(
        self,
        domain: MCDomain,
        config: MCConfig,
        importance_function: Callable[[NDArray], NDArray],
        importance_max: float | None = None,
    ):
        """
        Initialize importance sampling.

        Args:
            domain: Domain bounds
            config: MC configuration
            importance_function: Probability density p(x), normalized on the domain
            importance_max: An upper bound on p over the domain. If omitted, it is estimated from a dense draw.
        """
        super().__init__(domain, config)
        self.importance_function = importance_function
        check = np.random.RandomState(self._CHECK_SEED)
        lows, highs = self.domain_bounds[:, 0], self.domain_bounds[:, 1]
        points = lows + (highs - lows) * check.random_sample((self._CHECK_SAMPLES, self.dimension))
        values = np.asarray(importance_function(points), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            bad = int(np.flatnonzero(~np.isfinite(values) | (values < 0.0))[0])
            raise ValueError(
                f"ImportanceMCSampler: importance_function must be finite and non-negative on the domain; it is "
                f"{float(values[bad])!r} at x = {points[bad].tolist()}."
            )
        self.zero_density_points = points[values == 0.0]
        integral = float(self.domain_volume * np.mean(values))
        error = float(self.domain_volume * np.std(values) / np.sqrt(len(values)))
        if abs(integral - 1.0) > self.NORMALIZATION_TOL and abs(integral - 1.0) > 4.0 * error:
            raise ValueError(
                f"ImportanceMCSampler: importance_function must be a probability density normalized on the "
                f"domain; its integral is estimated at {integral:.6g} +- {error:.2g} ({self._CHECK_SAMPLES} "
                f"fixed-seed points), not 1. Divide it by its integral over the domain."
            )
        if importance_max is not None and float(np.max(values)) > float(importance_max):
            top = int(np.argmax(values))
            raise ValueError(
                f"ImportanceMCSampler: p(x) = {values[top]:.6g} at x = {points[top].tolist()} exceeds the rejection "
                f"bound M = {float(importance_max):.6g} (importance_max), so the draw would not follow p. Pass an "
                f"importance_max at least the density's supremum."
            )
        self.importance_max = (
            float(importance_max) if importance_max is not None else float(np.max(values)) * self.ENVELOPE_SAFETY
        )
        acceptance = 1.0 / (self.domain_volume * self.importance_max)
        if acceptance < self.MIN_ACCEPTANCE:
            raise ValueError(
                f"ImportanceMCSampler: the rejection acceptance rate 1/(V M) = {acceptance:.3g} is below "
                f"{self.MIN_ACCEPTANCE:g} (V = {self.domain_volume:.6g}, M = {self.importance_max:.6g}): the bound M on "
                f"the density, importance_max or the largest value over the check points, is too large for rejection "
                f"from this domain. A wider density, a smaller domain, or a tighter importance_max raises the rate."
            )

    def sample(self, num_samples: int) -> NDArray:
        """Draw exactly ``num_samples`` points from the density by rejection, with no uniform fill."""
        lows, highs = self.domain_bounds[:, 0], self.domain_bounds[:, 1]
        accepted: list[NDArray] = []
        count = 0
        while count < num_samples:
            batch = max(4 * (num_samples - count), 64)
            candidates = lows + (highs - lows) * self.rng.random_sample((batch, self.dimension))
            values = np.asarray(self.importance_function(candidates), dtype=float)
            over = values > self.importance_max
            if np.any(over):
                i = int(np.argmax(over))
                raise ValueError(
                    f"ImportanceMCSampler: p(x) = {values[i]:.6g} at x = {candidates[i].tolist()} exceeds the "
                    f"rejection bound M = {self.importance_max:.6g}, so the draw would not follow p. Pass an "
                    f"importance_max at least the density's supremum."
                )
            keep = candidates[self.rng.random_sample(batch) < values / self.importance_max]
            accepted.append(keep)
            count += len(keep)
        return np.vstack(accepted)[:num_samples]

    def get_weights(self, points: NDArray) -> NDArray:
        """Weights 1 / (N V p(x)), so that ``monte_carlo_integrate``'s estimate is mean(f / p)."""
        values = np.asarray(self.importance_function(points), dtype=float)
        return 1.0 / (len(points) * self.domain_volume * values)


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


#: The ``sampling_method`` values ``monte_carlo_integrate`` dispatches; any other is refused (#2648, ruling 8).
SAMPLING_METHODS = ("uniform", "stratified", "sobol", "halton", "latin_hypercube", "importance")


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
    elif config.sampling_method == "importance":
        if config.importance_function is None:
            raise ValueError(
                "monte_carlo_integrate: sampling_method='importance' needs MCConfig.importance_function, a "
                "probability density normalized on the domain."
            )
        if config.use_control_variates:
            raise ValueError(
                "monte_carlo_integrate: use_control_variates does not combine with sampling_method='importance': the "
                "control variate is calibrated on its sample mean with equal weights, which importance weights break, "
                "and the estimate comes back biased (#2648). Use one or the other."
            )
        sampler = ImportanceMCSampler(domain, config, config.importance_function)
    else:
        raise ValueError(
            f"monte_carlo_integrate: unknown sampling_method {config.sampling_method!r}; choose one of "
            f"{', '.join(repr(m) for m in SAMPLING_METHODS)}. (It used to be logged and sampled uniformly, #2648.)"
        )

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

    # Each point's contribution to the estimate: V f(x) w(x) N, whose mean is the estimate
    domain_volume = np.prod([max_val - min_val for min_val, max_val in domain])
    contributions = function_values * weights * len(weights)
    if isinstance(sampler, ImportanceMCSampler):
        _refuse_integrand_outside_the_support(integrand, sampler, contributions, domain_volume)
    return _mc_result(contributions, config, start_time, scale=domain_volume)


def _mc_result(contributions: NDArray, config: MCConfig, start_time: float, scale: float = 1.0) -> MCResult:
    """The estimate, its standard error and confidence interval from per-point contributions: the estimate is
    ``scale`` times their mean. The one place an ``MCResult`` is assembled from a draw (``monte_carlo_integrate``,
    ``integrate_gaussian_quadrature_mc``); ``num_samples_used`` is the number of points drawn."""
    import time

    estimate = scale * np.mean(contributions)
    standard_error = scale * np.std(contributions) / np.sqrt(len(contributions))

    # Confidence interval
    z_score = 1.96 if config.confidence_level == 0.95 else 2.576  # Simplified
    ci_lower = estimate - z_score * standard_error
    ci_upper = estimate + z_score * standard_error

    elapsed = time.time() - start_time
    return MCResult(
        estimate=estimate,
        standard_error=standard_error,
        confidence_interval=(ci_lower, ci_upper),
        num_samples_used=len(contributions),
        effective_sample_size=len(contributions),
        converged=standard_error < config.error_tolerance,
        final_error=standard_error,
        computation_time=elapsed,
        samples_per_second=len(contributions) / max(elapsed, 1e-6),
    )


def _refuse_integrand_outside_the_support(
    integrand: MCIntegrand, sampler: ImportanceMCSampler, contributions: NDArray, scale: float
) -> None:
    """Importance sampling misses the integral of f wherever its density vanishes. The part of the integral of |f| it
    cannot see is estimated on the sampler's check points where p = 0, the only points ``integrand`` is evaluated at
    here; the part it sees is the draw's own estimate of the integral of |f| over p > 0 (``scale`` times the mean
    |contribution|). A missing fraction above ``ImportanceMCSampler.NORMALIZATION_TOL`` is refused, naming the
    fraction and the check point where |f| is largest -- a mass criterion, so an f that is merely subnormal where p
    underflows to 0 is not refused (review 2 of #2664). Sampled: a miss between check points is not seen (#2648)."""
    points = sampler.zero_density_points
    if len(points) == 0:
        return
    values = np.abs(np.asarray(integrand(points), dtype=float))
    missing = sampler.domain_volume * float(np.sum(values)) / sampler._CHECK_SAMPLES
    seen = scale * float(np.mean(np.abs(contributions)))
    if missing > sampler.NORMALIZATION_TOL * (missing + seen):
        i = int(np.argmax(values))
        raise ValueError(
            f"monte_carlo_integrate: an estimated {missing / (missing + seen):.3g} of the integral of |f| lies where "
            f"the importance density is 0 (at x = {points[i].tolist()} the integrand is {float(values[i])!r}); "
            "importance sampling never draws there, so that part of the integral would be missing. Use a density "
            f"positive wherever the integrand is non-zero (the missing fraction allowed is "
            f"{sampler.NORMALIZATION_TOL:g})."
        )


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
    mean: NDArray, cov: NDArray, integrand: MCIntegrand, num_samples: int = 10000, seed: int | None = None
) -> MCResult:
    """The integral of ``integrand`` over all of R^d, by importance sampling from N(mean, cov) (#2648, ruling 8).

    This is not the Gaussian-weighted expectation E_N[f] that Gauss-Hermite quadrature computes. The points are
    drawn from N(mean, cov) directly and the estimate is mean(f(x) / p(x)), p the normal density, unbiased for the
    integral of f over R^d. Its variance is finite only when the integral of f^2 / p is: an integrand with heavier
    tails than the Gaussian, or one that is not integrable (f = 1), gives an estimate whose standard error means
    nothing. f / p is formed in log space and rescaled by its largest value in the draw before averaging, so neither
    p nor 1 / p overflows on its own (review 2 of #2664); the estimate overflows only when that largest f / p does.

    ``cov`` must be symmetric positive definite, and is refused by name otherwise. ``seed`` fixes the draw.
    """
    import time

    start_time = time.time()
    config = MCConfig(num_samples=num_samples, seed=seed)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (len(mean), len(mean)) or not np.allclose(cov, cov.T, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"integrate_gaussian_quadrature_mc: cov must be a symmetric {len(mean)}x{len(mean)} matrix; a Cholesky "
            "factor would read only its lower triangle."
        )
    # Cholesky, not inv() (Issue #1066): stable for an ill-conditioned covariance, and its diagonal gives log det.
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "integrate_gaussian_quadrature_mc: cov must be positive definite; it has no Cholesky factor."
        ) from exc
    z = np.random.RandomState(config.seed).standard_normal((num_samples, len(mean)))
    points = mean + z @ factor.T
    log_density = -0.5 * np.sum(z**2, axis=1) - 0.5 * len(mean) * np.log(2.0 * np.pi) - np.sum(np.log(np.diag(factor)))
    values = np.asarray(integrand(points), dtype=float)
    with np.errstate(divide="ignore"):
        log_ratio = np.log(np.abs(values)) - log_density
    finite = log_ratio[np.isfinite(log_ratio)]
    shift = float(np.max(finite)) if finite.size else 0.0
    contributions = np.sign(values) * np.exp(log_ratio - shift)
    return _mc_result(contributions, config, start_time, scale=float(np.exp(shift)))


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
