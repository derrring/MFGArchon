"""
Boundary-corrected KDE for particle density estimation (Issue #709).

This module provides KDE methods that correctly handle domain boundaries,
avoiding the ~50% density underestimation at boundaries that occurs with
standard KDE.

Three methods are provided:

1. **Reflection KDE** (Schuster 1985): the method of images. The estimate adds
   the sample's KDE at each image of the evaluation point across nearby walls.

2. **Renormalization KDE**: Normalizes truncated kernel at each evaluation
   point. More uniform near boundaries but computationally more expensive.

3. **Beta KDE** (Chen 1999): Uses Beta distribution kernels that adapt shape
   based on distance from boundaries. Theoretically optimal, no ghost particles
   needed, but 1D only.

Example:
    >>> from mfgarchon.utils.numerical.particle import reflection_kde, beta_kde
    >>> particles = np.random.uniform(-0.5, 0.5, 5000)
    >>> x_eval = np.linspace(-0.5, 0.5, 51)
    >>> # Reflection method (works in nD)
    >>> density_ref = reflection_kde(particles, x_eval, bandwidth="scott",
    ...                              bounds=[(-0.5, 0.5)])
    >>> # Beta kernel method (1D only, most accurate)
    >>> density_beta = beta_kde(particles, x_eval, bandwidth=0.05,
    ...                         bounds=(-0.5, 0.5))

Reference:
    - Chen (1999). "Beta kernel estimators for density functions"
    - Karunamuni & Alberts (2005). "On boundary correction in kernel density estimation"
    - Schuster (1985). "Incorporating support constraints into nonparametric estimators"

Issue #709: FPParticleSolver boundary bias fix
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Try to import scipy
try:
    from scipy.special import beta as beta_func
    from scipy.stats import beta as beta_dist
    from scipy.stats import gaussian_kde

    SCIPY_AVAILABLE = True
except ImportError:
    gaussian_kde = None
    beta_func = None
    beta_dist = None
    SCIPY_AVAILABLE = False


def create_ghost_particles(
    particles: NDArray[np.floating],
    bounds: list[tuple[float, float]],
    bandwidth: float | NDArray[np.floating],
    n_bandwidths: float = 3.0,
) -> NDArray[np.floating]:
    """
    Create ghost (mirror) particles for reflection KDE (dimension-agnostic).

    For each particle near boundaries, create all necessary reflections
    including face, edge, and corner reflections. This requires up to
    2^k reflections for particles near k boundaries.

    Args:
        particles: Particle positions, shape (N,) for 1D or (N, d) for nD
        bounds: Domain bounds [(xmin, xmax), ...] - length determines dimension
        bandwidth: KDE bandwidth - scalar or per-dimension array
        n_bandwidths: Number of bandwidths for reflection zone (default: 3.0)

    Returns:
        Ghost particles with same shape convention as input:
        - 1D input (N,) -> output (M,)
        - nD input (N, d) -> output (M, d)

    Example (1D):
        >>> particles = np.array([0.05, 0.1, 0.5, 0.9, 0.95])
        >>> ghosts = create_ghost_particles(particles, [(0.0, 1.0)], bandwidth=0.1)
        >>> # ghosts = [-0.05, -0.1, 1.1, 1.05]

    Example (2D corner):
        >>> particles = np.array([[0.05, 0.05]])  # Near corner
        >>> ghosts = create_ghost_particles(particles, [(0, 1), (0, 1)], bandwidth=0.1)
        >>> # Creates 3 ghosts: (-0.05, 0.05), (0.05, -0.05), (-0.05, -0.05)
    """
    # Normalize to 2D array for unified processing
    particles = np.atleast_1d(particles)
    is_1d = particles.ndim == 1
    if is_1d:
        particles = particles.reshape(-1, 1)

    ndim = len(bounds)

    # Handle scalar bandwidth
    if np.isscalar(bandwidth):
        bandwidth_arr = np.full(ndim, bandwidth)
    else:
        bandwidth_arr = np.asarray(bandwidth)

    ghost_list = []

    # For each particle, determine which boundaries it's near
    # and create all 2^k reflections (where k = number of nearby boundaries)
    for p in particles:
        # Find which boundaries this particle is near
        # Each dimension can be: 0=no reflection, -1=reflect left, +1=reflect right
        boundary_status = []
        for d in range(ndim):
            xmin, xmax = bounds[d]
            bw_d = bandwidth_arr[d] if d < len(bandwidth_arr) else bandwidth_arr[0]
            reflection_zone = n_bandwidths * bw_d

            status = [0]  # Always include "no reflection" option
            if p[d] < xmin + reflection_zone:
                status.append(-1)  # Near left boundary
            if p[d] > xmax - reflection_zone:
                status.append(+1)  # Near right boundary
            boundary_status.append(status)

        # Generate all combinations of reflections
        for combo in product(*boundary_status):
            # Skip the identity (no reflection at all)
            if all(s == 0 for s in combo):
                continue

            # Create reflected particle
            ghost = p.copy()
            for d, s in enumerate(combo):
                if s == -1:  # Reflect across left boundary
                    ghost[d] = 2 * bounds[d][0] - ghost[d]
                elif s == +1:  # Reflect across right boundary
                    ghost[d] = 2 * bounds[d][1] - ghost[d]

            ghost_list.append(ghost)

    # Return with same shape convention as input
    if ghost_list:
        ghosts = np.array(ghost_list, dtype=particles.dtype)
        return ghosts.ravel() if is_1d else ghosts
    else:
        return np.array([], dtype=particles.dtype) if is_1d else np.empty((0, ndim), dtype=particles.dtype)


def reflection_kde(
    particles: NDArray[np.floating],
    eval_points: NDArray[np.floating],
    bandwidth: float | str,
    bounds: list[tuple[float, float]],
    n_bandwidths: float = 3.0,
    normalize: bool = True,
) -> NDArray[np.floating]:
    """
    Boundary-corrected KDE using the reflection method (dimension-agnostic).

    The method of images: the estimate at x is the sample's Gaussian KDE at x, plus the same KDE at each image of x
    across the walls x is near (within ``n_bandwidths`` kernel widths, on every reflected axis; at a corner, every
    combination). A particle's image across a wall carries the reflected kernel R Sigma R, which is what evaluating the
    sample's own kernel Sigma at the reflected point gives, so the estimate's derivative normal to a wall is zero on the
    wall for any sample covariance (#2556), for a kernel narrower than the domain: there is one image per wall, and a
    kernel that reaches the far wall would need the images' images. The kernel is the real sample's, as
    `scipy.stats.gaussian_kde` builds it for ``bandwidth`` (`kde_kernel_covariance`, CONVENTIONS.md § 9).

    This eliminates the ~50% density underestimation at boundaries
    that occurs with standard KDE.

    Args:
        particles: Particle positions, shape (N,) for 1D or (N, d) for nD
        eval_points: Points at which to evaluate density
            - 1D: shape (M,)
            - nD: shape (M, d)
        bandwidth: KDE bandwidth: a positive factor of the sample's standard deviation, or 'scott' or 'silverman'
            (`kde_factor`)
        bounds: Domain bounds [(xmin, xmax), ...] - length determines dimension
        n_bandwidths: Kernel widths from a wall within which a point's image across it is added (default: 3.0)
        normalize: If True (default), normalize density to integrate to 1 using
            grid-based quadrature (assumes eval_points form a regular grid).
            Set to False for scattered/meshfree points and normalize externally
            using appropriate integration weights (e.g., Voronoi volumes).

    Returns:
        Density estimates at eval_points, shape (M,)

    Raises:
        ValueError: for a bandwidth that is not a positive factor or a rule, and for a sample with no spread
            (`kde_kernel_covariance`).

    Example (1D):
        >>> particles = np.random.uniform(-0.5, 0.5, 5000)
        >>> x_eval = np.linspace(-0.5, 0.5, 51)
        >>> density = reflection_kde(particles, x_eval, bandwidth="scott",
        ...                          bounds=[(-0.5, 0.5)])

    Example (2D, regular grid):
        >>> particles = np.random.uniform([0, 0], [1, 1], (2000, 2))
        >>> grid = np.column_stack([XX.ravel(), YY.ravel()])
        >>> density = reflection_kde(particles, grid, bandwidth="scott",
        ...                          bounds=[(0, 1), (0, 1)])

    Example (2D, scattered points with Voronoi normalization):
        >>> raw_kde = reflection_kde(particles, scattered_pts, bandwidth="scott",
        ...                          bounds=[(0, 1), (0, 1)], normalize=False)
        >>> density = raw_kde / np.sum(raw_kde * voronoi_volumes)

    Note:
        For periodic BC, use standard KDE (no boundary bias).
        For Dirichlet/absorbing BC, this method may not be appropriate.
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for KDE. Install with: pip install scipy")
    from mfgarchon.utils.numerical.particle.interpolation import kde_factor, kde_kernel_covariance

    # Float, so that an integer evaluation point's image across a wall is not truncated
    particles = np.atleast_1d(np.asarray(particles, dtype=float))
    eval_points = np.atleast_1d(np.asarray(eval_points, dtype=float))
    is_1d = particles.ndim == 1
    ndim = len(bounds)
    if is_1d:
        particles = particles.reshape(-1, 1)
        eval_points = eval_points.reshape(-1, 1)

    # The real sample's kernel (refusing a sample with no spread by name), and its per-axis widths for the zone.
    widths = np.sqrt(np.diag(kde_kernel_covariance(particles, bandwidth)))
    kde = gaussian_kde(particles.T, bw_method=kde_factor(bandwidth, particles))
    zone = n_bandwidths * widths

    density = kde(eval_points.T)
    # Each image of a point across the walls it is near: on each axis none, the lower wall, or the upper wall.
    for pattern in product((None, 0, 1), repeat=ndim):
        if all(side is None for side in pattern):
            continue
        image = eval_points.copy()
        near = np.ones(len(eval_points), dtype=bool)
        for d, side in enumerate(pattern):
            if side is None:
                continue
            wall = bounds[d][side]
            near &= np.abs(eval_points[:, d] - wall) < zone[d]
            image[:, d] = 2.0 * wall - eval_points[:, d]
        if np.any(near):
            density[near] += kde(image[near].T)

    for d in range(ndim):
        xmin, xmax = bounds[d]
        outside_mask = (eval_points[:, d] < xmin) | (eval_points[:, d] > xmax)
        density[outside_mask] = 0

    if normalize and eval_points.shape[0] > 1:
        dV = 1.0
        for d in range(ndim):
            unique_coords = np.unique(eval_points[:, d])
            if len(unique_coords) > 1:
                dV *= (unique_coords[-1] - unique_coords[0]) / (len(unique_coords) - 1)
        total_mass = np.sum(density) * dV
        if total_mass > 1e-10:
            density = density / total_mass

    return density


def beta_kde(
    particles: NDArray[np.floating],
    eval_points: NDArray[np.floating],
    bandwidth: float | NDArray[np.floating],
    bounds: tuple[float, float] | list[tuple[float, float]],
) -> NDArray[np.floating]:
    """
    Boundary-corrected KDE using Beta kernels (Chen 1999), dimension-agnostic.

    Beta kernels naturally adapt their shape based on distance from boundaries:
    - At boundaries: asymmetric kernel, all mass inside domain
    - At interior: symmetric kernel, similar to Gaussian

    This method achieves optimal MSE = O(h^4) even at boundaries, without
    ghost particles or normalization factors.

    For nD, uses product kernel: K(t; x) = prod_d K_Beta(t_d; a_d, b_d)

    Args:
        particles: Particle positions, shape (N,) for 1D or (N, d) for nD
        eval_points: Points at which to evaluate density
            - 1D: shape (M,)
            - nD: shape (M, d)
        bandwidth: KDE bandwidth - scalar or per-dimension array
        bounds: Domain bounds
            - 1D: tuple (xmin, xmax)
            - nD: list [(xmin, xmax), (ymin, ymax), ...]

    Returns:
        Density estimates at eval_points, shape (M,)

    Mathematical formulation:
        For data on [0,1], the Beta kernel estimator is:

        f_hat(x) = (1/n) * sum_i K_Beta(X_i; a(x), b(x))

        where:
        - a(x) = x/h + 1
        - b(x) = (1-x)/h + 1
        - K_Beta(t; a, b) = t^(a-1) * (1-t)^(b-1) / B(a, b)

        For nD, use product kernel:
        K(t; x) = prod_d K_Beta(t_d; a_d(x_d), b_d(x_d))

    Example (1D):
        >>> particles = np.random.uniform(-0.5, 0.5, 5000)
        >>> x_eval = np.linspace(-0.5, 0.5, 51)
        >>> density = beta_kde(particles, x_eval, bandwidth=0.05,
        ...                    bounds=(-0.5, 0.5))

    Example (2D):
        >>> particles = np.random.uniform([0, 0], [1, 1], (2000, 2))
        >>> grid = np.column_stack([XX.ravel(), YY.ravel()])
        >>> density = beta_kde(particles, grid, bandwidth=0.1,
        ...                    bounds=[(0, 1), (0, 1)])

    Reference:
        Chen, S. X. (1999). "Beta kernel estimators for density functions."
        Computational Statistics & Data Analysis, 31(2), 131-145.

    Note:
        - Optimal for bounded support; naturally handles boundaries.
        - Bandwidth h should be chosen appropriately (typically 0.01-0.1).
        - For nD, product kernel assumes approximate independence between dimensions.
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for Beta KDE. Install with: pip install scipy")

    # Normalize input arrays
    particles = np.atleast_1d(particles)
    eval_points = np.atleast_1d(eval_points)

    # Detect 1D vs nD from bounds format
    if isinstance(bounds, tuple) and len(bounds) == 2 and not isinstance(bounds[0], tuple):
        # 1D case: bounds = (xmin, xmax)
        bounds = [bounds]
        is_1d = True
    else:
        # nD case: bounds = [(xmin, xmax), ...]
        is_1d = particles.ndim == 1

    ndim = len(bounds)

    if is_1d:
        particles = particles.reshape(-1, 1)
        eval_points = eval_points.reshape(-1, 1)

    n = len(particles)
    m = len(eval_points)

    # Handle bandwidth - scalar or per-dimension
    if np.isscalar(bandwidth):
        bandwidth_arr = np.full(ndim, bandwidth)
    else:
        bandwidth_arr = np.asarray(bandwidth)

    # Transform particles to [0, 1]^d
    particles_01 = np.zeros_like(particles)
    domain_lengths = np.zeros(ndim)
    for d in range(ndim):
        xmin, xmax = bounds[d]
        domain_lengths[d] = xmax - xmin
        particles_01[:, d] = (particles[:, d] - xmin) / domain_lengths[d]

    # Clip to valid range (numerical safety)
    particles_01 = np.clip(particles_01, 1e-10, 1 - 1e-10)

    # Scaled bandwidth for [0,1] domain
    h_scaled = bandwidth_arr / domain_lengths

    density = np.zeros(m)

    for j in range(m):
        x = eval_points[j]

        # Transform evaluation point to [0, 1]^d
        x_01 = np.zeros(ndim)
        outside = False
        for d in range(ndim):
            xmin, xmax = bounds[d]
            x_01[d] = (x[d] - xmin) / domain_lengths[d]
            if x_01[d] < 0 or x_01[d] > 1:
                outside = True
                break

        if outside:
            density[j] = 0
            continue

        # Product kernel: K(t; x) = prod_d K_Beta(t_d; a_d, b_d)
        # Start with all ones, multiply by each dimension's contribution
        kernel_values = np.ones(n)

        for d in range(ndim):
            h_d = h_scaled[d]
            # Beta kernel parameters (Chen 1999, Eq. 2.1)
            a_d = x_01[d] / h_d + 1
            b_d = (1 - x_01[d]) / h_d + 1

            # Evaluate Beta(a, b) PDF at each particle position for this dimension
            kernel_values *= beta_dist.pdf(particles_01[:, d], a_d, b_d)

        # Average over all particles
        density[j] = np.mean(kernel_values)

    # Scale back to original domain (product of domain lengths for nD)
    total_domain_volume = np.prod(domain_lengths)
    density = density / total_domain_volume

    return density


def renormalization_kde(
    particles: NDArray[np.floating],
    eval_points: NDArray[np.floating],
    bandwidth: float | str,
    bounds: list[tuple[float, float]],
) -> NDArray[np.floating]:
    """
    Boundary-corrected KDE using kernel renormalization (dimension-agnostic).

    At each evaluation point, the truncated kernel is renormalized so that
    it integrates to 1 over the valid domain. This eliminates boundary bias
    without creating ghost particles.

    Args:
        particles: Particle positions, shape (N,) for 1D or (N, d) for nD
        eval_points: Points at which to evaluate density
        bandwidth: KDE bandwidth: a positive factor of the sample's standard deviation, or 'scott' or
            'silverman' (`kde_factor`); the renormalization reads the same kernel's per-axis widths
        bounds: Domain bounds [(xmin, xmax), ...] - length determines dimension

    Returns:
        Density estimates at eval_points, shape (M,)

    Mathematical formulation:
        f_hat(x) = (1/n) * sum_i K_h(x - X_i) / integral_D K_h(x - y) dy

        The normalization factor at x is computed using the Gaussian CDF:
        Z(x) = prod_d [Phi((x_d - L_d)/h) - Phi((x_d - U_d)/h)]

    Note:
        More computationally expensive than reflection KDE due to
        per-point normalization, but provides more uniform estimates.
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for KDE. Install with: pip install scipy")

    from scipy.stats import norm

    # Normalize arrays
    particles = np.atleast_1d(particles)
    eval_points = np.atleast_1d(eval_points)
    is_1d = particles.ndim == 1
    ndim = len(bounds)

    if is_1d:
        particles = particles.reshape(-1, 1)
        eval_points = eval_points.reshape(-1, 1)

    m = len(eval_points)

    # The real sample's kernel, as scipy builds it for `bandwidth` (CONVENTIONS.md § 9; #2556).
    from mfgarchon.utils.numerical.particle.interpolation import kde_factor, kde_kernel_covariance

    widths = np.sqrt(np.diag(kde_kernel_covariance(particles, bandwidth)))
    kde = gaussian_kde(particles.T, bw_method=kde_factor(bandwidth, particles))

    # Get raw density values
    density_raw = kde(eval_points.T)

    # Compute normalization factors for each eval point
    # Z(x) = product over dimensions of [Phi((U-x)/h_d) - Phi((L-x)/h_d)], h_d the kernel's width on axis d
    normalization = np.ones(m)

    for d in range(ndim):
        xmin, xmax = bounds[d]
        # For each eval point, compute the integral of Gaussian kernel over [xmin, xmax]
        # Integral = Phi((xmax - x)/h_d) - Phi((xmin - x)/h_d)
        z_right = norm.cdf((xmax - eval_points[:, d]) / widths[d])
        z_left = norm.cdf((xmin - eval_points[:, d]) / widths[d])
        normalization *= z_right - z_left

    # Avoid division by zero at boundaries
    normalization = np.maximum(normalization, 1e-10)

    # Renormalized density
    density = density_raw / normalization

    # Clip points outside domain to zero
    for d in range(ndim):
        xmin, xmax = bounds[d]
        outside_mask = (eval_points[:, d] < xmin) | (eval_points[:, d] > xmax)
        density[outside_mask] = 0

    # Domain normalization: ensure density integrates to 1 over the domain.
    # Kernel renormalization corrects boundary bias but may not perfectly
    # preserve total mass. Explicit normalization fixes residual error. (Issue #718)
    if eval_points.shape[0] > 1:
        dV = 1.0
        for d in range(ndim):
            unique_coords = np.unique(eval_points[:, d])
            if len(unique_coords) > 1:
                dV *= (unique_coords[-1] - unique_coords[0]) / (len(unique_coords) - 1)
        total_mass = np.sum(density) * dV
        if total_mass > 1e-10:
            density = density / total_mass

    return density


# =============================================================================
# Legacy aliases for backward compatibility
# =============================================================================


def create_ghost_particles_1d(
    particles: NDArray[np.floating],
    xmin: float,
    xmax: float,
    bandwidth: float,
    n_bandwidths: float = 3.0,
) -> NDArray[np.floating]:
    """Legacy 1D alias. Use create_ghost_particles() instead."""
    return create_ghost_particles(particles, [(xmin, xmax)], bandwidth, n_bandwidths)


def reflection_kde_1d(
    particles: NDArray[np.floating],
    x_eval: NDArray[np.floating],
    bandwidth: float | str,
    xmin: float,
    xmax: float,
    n_bandwidths: float = 3.0,
) -> NDArray[np.floating]:
    """Legacy 1D alias. Use reflection_kde() instead."""
    return reflection_kde(particles, x_eval, bandwidth, [(xmin, xmax)], n_bandwidths)


def create_ghost_particles_nd(
    particles: NDArray[np.floating],
    bounds: list[tuple[float, float]],
    bandwidth: float | NDArray[np.floating],
    n_bandwidths: float = 3.0,
) -> NDArray[np.floating]:
    """Legacy nD alias. Use create_ghost_particles() instead."""
    return create_ghost_particles(particles, bounds, bandwidth, n_bandwidths)


def reflection_kde_nd(
    particles: NDArray[np.floating],
    grid_points: NDArray[np.floating],
    bandwidth: float | str | NDArray[np.floating],
    bounds: list[tuple[float, float]],
    n_bandwidths: float = 3.0,
) -> NDArray[np.floating]:
    """Legacy nD alias. Use reflection_kde() instead."""
    return reflection_kde(particles, grid_points, bandwidth, bounds, n_bandwidths)
