"""
Dimension-agnostic density estimation for particle-based FP solvers (Issue #635).

This module provides unified functions that work for any dimension:
- Density normalization
- KDE density estimation (particles → grid)
- Gradient computation
- Brownian increment generation
- Particle sampling from density

All functions accept both 1D and nD inputs with consistent APIs.
Internal optimizations (e.g., scipy fast path for 1D) are implementation details.

Usage:
    from mfgarchon.alg.numerical.fp_solvers.fp_particle_density import (
        normalize_density,
        estimate_density_kde,
        compute_gradient,
        generate_brownian_increment,
        sample_particles_from_density,
    )
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

# Optional scipy for KDE
try:  # pragma: no cover - optional SciPy dependency
    from scipy.stats import gaussian_kde as scipy_gaussian_kde

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful fallback when SciPy missing
    scipy_gaussian_kde = None
    SCIPY_AVAILABLE = False

# Issue #625: Migrated from tensor_calculus to operators/stencils
from mfgarchon.operators.stencils.finite_difference import gradient_nd
from mfgarchon.utils.numerical.particle import (
    interpolate_particles_to_grid,
    sample_from_density,
)

# =============================================================================
# Density Normalization (unified)
# =============================================================================


def normalize_density(
    density: np.ndarray,
    spacings: float | list[float] | tuple[float, ...],
) -> np.ndarray:
    """
    Normalize density to integrate to 1 (dimension-agnostic).

    Parameters
    ----------
    density : np.ndarray
        Density values on grid, shape (N1,) for 1D or (N1, N2, ..., Nd) for nD
    spacings : float or list[float]
        Grid spacing(s). Single float for 1D, list for nD.

    Returns
    -------
    normalized : np.ndarray
        Normalized density (integrates to 1)

    Examples
    --------
    >>> density_1d = np.array([0.5, 1.0, 0.5])
    >>> normalize_density(density_1d, 0.1)  # 1D
    >>> density_2d = np.random.rand(20, 20)
    >>> normalize_density(density_2d, [0.05, 0.05])  # 2D
    """
    # Normalize spacings to list
    if isinstance(spacings, (int, float)):
        spacings = [float(spacings)]
    else:
        spacings = list(spacings)

    # Volume element = product of all spacings
    dV = float(np.prod(spacings))

    if dV < 1e-14:
        # Degenerate case: zero volume element
        sum_val = np.sum(density)
        if sum_val > 1e-9:
            return density / sum_val
        return density

    total_mass = np.sum(density) * dV
    if total_mass > 1e-14:
        return density / total_mass
    return density


# =============================================================================
# Gradient Computation (unified)
# =============================================================================


def compute_gradient(
    grid_values: np.ndarray,
    spacings: float | list[float] | tuple[float, ...],
) -> np.ndarray:
    """
    Compute gradient of a scalar field on a grid (dimension-agnostic).

    Uses central differences via stencils.gradient_nd utility.

    Parameters
    ----------
    grid_values : np.ndarray
        Scalar field values on grid, shape (N1,) for 1D or (N1, N2, ..., Nd) for nD
    spacings : float or list[float]
        Grid spacing(s). Single float for 1D, list for nD.

    Returns
    -------
    gradient : np.ndarray
        Gradient field.
        - 1D: shape (N1,) - single component
        - nD: shape (N1, N2, ..., Nd, d) - last axis is gradient components

    Examples
    --------
    >>> field_1d = np.linspace(0, 1, 50)**2
    >>> grad_1d = compute_gradient(field_1d, 0.02)  # shape (50,)
    >>> field_2d = X**2 + Y**2  # meshgrid
    >>> grad_2d = compute_gradient(field_2d, [0.05, 0.05])  # shape (Nx, Ny, 2)

    Note:
        Issue #625: Migrated from tensor_calculus.gradient_simple to stencils.gradient_nd
    """
    # Normalize spacings to list
    if isinstance(spacings, (int, float)):
        spacings = [float(spacings)]
    else:
        spacings = list(spacings)

    # gradient_nd returns list of arrays (one per dimension)
    grad_components = gradient_nd(grid_values, spacings)

    if len(grad_components) == 1:
        # 1D: return flat array (backward compatible)
        return grad_components[0]

    # nD: stack into single array with last axis = dimension
    return np.stack(grad_components, axis=-1)


# =============================================================================
# Brownian Increment Generation (unified)
# =============================================================================


def generate_brownian_increment(
    num_particles: int,
    dimension: int,
    dt: float,
    sigma: float,
    rng: Any = np.random,
) -> np.ndarray:
    """
    Generate Brownian increment for SDE evolution (dimension-agnostic).

    The increment follows dX = sigma * dW where dW ~ N(0, sqrt(dt)).

    Parameters
    ----------
    num_particles : int
        Number of particles
    dimension : int
        Spatial dimension (1, 2, 3, ...)
    dt : float
        Time step size
    sigma : float
        Diffusion coefficient

    Returns
    -------
    dW : np.ndarray
        Brownian increments.
        - 1D: shape (num_particles,)
        - nD: shape (num_particles, dimension)

    Examples
    --------
    >>> dW_1d = generate_brownian_increment(1000, 1, 0.01, 0.1)  # (1000,)
    >>> dW_2d = generate_brownian_increment(1000, 2, 0.01, 0.1)  # (1000, 2)
    """
    if dt < 1e-14:
        if dimension == 1:
            return np.zeros(num_particles)
        return np.zeros((num_particles, dimension))

    # Independent Brownian motion in each dimension
    if dimension == 1:
        return sigma * rng.normal(0, np.sqrt(dt), num_particles)

    return sigma * rng.normal(0, np.sqrt(dt), (num_particles, dimension))


# =============================================================================
# KDE Density Estimation (unified)
# =============================================================================


def estimate_density_kde(
    particles: np.ndarray,
    grid_shape: tuple[int, ...],
    bounds: tuple[tuple[float, float], ...],
    kde_bandwidth: Any = "scott",
    backend: str | None = None,
    coordinates: np.ndarray | list[np.ndarray] | None = None,
) -> np.ndarray:
    """
    Estimate density from particles using KDE (dimension-agnostic).

    Automatically selects optimal implementation based on dimension and backend:
    - 1D + CPU: scipy.stats.gaussian_kde (fast, mature)
    - 1D + GPU: gaussian_kde_gpu
    - nD: interpolate_particles_to_grid utility

    Parameters
    ----------
    particles : np.ndarray
        Particle positions.
        - 1D: shape (num_particles,) or (num_particles, 1)
        - nD: shape (num_particles, dimension)
    grid_shape : tuple[int, ...]
        Grid shape, e.g., (Nx,) for 1D, (Nx, Ny) for 2D
    bounds : tuple[tuple[float, float], ...]
        Bounds per dimension, e.g., ((xmin, xmax),) for 1D
    kde_bandwidth : Any
        Bandwidth for KDE. Options:
        - "scott" or "silverman": automatic bandwidth selection
        - float: fixed bandwidth value
    backend : str, optional
        Backend for GPU acceleration (e.g., "jax", "torch").
        Currently only affects 1D; nD GPU support planned.
    coordinates : np.ndarray or list[np.ndarray], optional
        Grid coordinates for evaluation. If None, computed from bounds/shape.

    Returns
    -------
    density : np.ndarray
        Density on grid, shape matches grid_shape

    Examples
    --------
    >>> particles_1d = np.random.normal(0.5, 0.1, 500)
    >>> density = estimate_density_kde(particles_1d, (50,), ((0, 1),))
    >>> particles_2d = np.random.normal(0.5, 0.1, (500, 2))
    >>> density = estimate_density_kde(particles_2d, (30, 30), ((0, 1), (0, 1)))
    """
    # Determine dimension from bounds
    dimension = len(bounds)

    # Normalize particles shape
    particles = np.asarray(particles)
    if particles.ndim == 1:
        # 1D particles as flat array
        if dimension != 1:
            raise ValueError(f"1D particles but {dimension}D bounds provided")
        particles_flat = particles
        particles_nd = particles[:, np.newaxis]
    else:
        particles_nd = particles
        particles_flat = particles.ravel() if dimension == 1 else None

    num_particles = len(particles_nd)

    # Edge case: no particles
    if num_particles == 0:
        return np.zeros(grid_shape)

    # Edge case: degenerate distribution (all at same location)
    unique_count = len(np.unique(particles_nd, axis=0))
    if unique_count < 2:
        return _handle_degenerate_particles(particles_nd, grid_shape, bounds, num_particles)

    # Route to optimal implementation
    if dimension == 1:
        return _estimate_density_kde_1d(
            particles_flat,
            grid_shape[0],
            bounds[0],
            kde_bandwidth,
            backend,
            coordinates[0] if coordinates is not None else None,
        )
    else:
        return _estimate_density_kde_nd(
            particles_nd,
            num_particles,
            grid_shape,
            bounds,
            kde_bandwidth,
        )


def _handle_degenerate_particles(
    particles: np.ndarray,
    grid_shape: tuple[int, ...],
    bounds: tuple[tuple[float, float], ...],
    num_particles: int,
) -> np.ndarray:
    """Handle edge case where all particles are at the same location."""
    dimension = len(bounds)
    density = np.zeros(grid_shape)
    mean_pos = np.mean(particles, axis=0)

    indices = []
    for d in range(dimension):
        coords = np.linspace(bounds[d][0], bounds[d][1], grid_shape[d])
        idx = np.argmin(np.abs(coords - mean_pos[d]))
        indices.append(idx)

    density[tuple(indices)] = num_particles
    return density


def _estimate_density_kde_1d(
    particles: np.ndarray,
    nx: int,
    bounds: tuple[float, float],
    kde_bandwidth: Any,
    backend: str | None,
    coordinates: np.ndarray | None,
) -> np.ndarray:
    """
    1D KDE implementation using scipy (CPU) or GPU backend.

    This is the optimized fast path for 1D problems.
    """
    xmin, xmax = bounds
    dx = (xmax - xmin) / (nx - 1) if nx > 1 else 1.0

    # Compute evaluation coordinates if not provided
    if coordinates is None:
        x_coords = np.linspace(xmin, xmax, nx)
    else:
        x_coords = coordinates

    # Check for degenerate case
    if np.std(particles) < 1e-9 * (xmax - xmin):
        density = np.zeros(nx)
        if len(particles) > 0:
            mean_pos = np.mean(particles)
            closest_idx = np.argmin(np.abs(x_coords - mean_pos))
            if dx > 1e-14:
                density[closest_idx] = 1.0 / dx
            elif nx == 1:
                density[closest_idx] = 1.0
        return density

    try:
        # GPU path
        if backend is not None:
            from mfgarchon.alg.numerical.density_estimation import gaussian_kde_gpu

            # Convert bandwidth to float if needed
            if isinstance(kde_bandwidth, str):
                from mfgarchon.alg.numerical.density_estimation import adaptive_bandwidth_selection

                bandwidth_value = adaptive_bandwidth_selection(particles, method=kde_bandwidth)
            else:
                bandwidth_value = float(kde_bandwidth)

            density = gaussian_kde_gpu(particles, x_coords, bandwidth_value, backend)

        # CPU path: scipy
        elif SCIPY_AVAILABLE and scipy_gaussian_kde is not None:
            kde = scipy_gaussian_kde(particles, bw_method=kde_bandwidth)
            density = kde(x_coords)

        else:
            raise RuntimeError("SciPy not available for KDE")

        # Clip to domain bounds
        density[x_coords < xmin] = 0
        density[x_coords > xmax] = 0

        return density

    except Exception as e:
        error_msg = (
            f"KDE density estimation failed: {e}\n"
            f"Number of particles: {len(particles)}\n"
            f"Grid size: {nx}\n"
            f"Bandwidth: {kde_bandwidth}\n"
            "Suggestions:\n"
            "  - Increase number of particles (Np > 100 recommended)\n"
            "  - Use fixed bandwidth: kde_bandwidth=0.1\n"
            "  - Check particle initialization and drift/diffusion"
        )
        raise RuntimeError(error_msg) from e


def _estimate_density_kde_nd(
    particles: np.ndarray,
    num_particles: int,
    grid_shape: tuple[int, ...],
    bounds: tuple[tuple[float, float], ...],
    kde_bandwidth: Any,
) -> np.ndarray:
    """
    nD KDE implementation using interpolate_particles_to_grid utility.

    Future: Will support GPU backends when nD KDE is implemented.
    """
    try:
        # Delegate to utils for KDE computation
        particle_values = np.ones(num_particles)

        density = interpolate_particles_to_grid(
            particle_values=particle_values,
            particle_positions=particles,
            grid_shape=grid_shape,
            grid_bounds=bounds,
            method="kde",
            bandwidth=kde_bandwidth,
        )

        return density

    except Exception as e:
        warnings.warn(f"KDE failed in nD: {e}. Returning histogram estimate.")
        # Fallback to histogram
        density, _ = np.histogramdd(
            particles,
            bins=list(grid_shape),
            range=list(bounds),
            density=True,
        )
        return density


# =============================================================================
# Particle Sampling from Density (unified)
# =============================================================================


def sample_particles_from_density(
    density: np.ndarray,
    num_particles: int,
    bounds: tuple[tuple[float, float], ...],
    implicit_domain: Any | None = None,
) -> np.ndarray:
    """
    Sample particles from a density distribution (dimension-agnostic).

    Uses importance sampling via sample_from_density utility.
    Supports implicit domains (obstacles) for valid region sampling.

    Parameters
    ----------
    density : np.ndarray
        Density on grid, shape (N1,) for 1D or (N1, N2, ..., Nd) for nD
    num_particles : int
        Number of particles to sample
    bounds : tuple[tuple[float, float], ...]
        Bounds per dimension, e.g., ((xmin, xmax),) for 1D
    implicit_domain : ImplicitDomain, optional
        If provided, only sample in valid regions (outside obstacles)

    Returns
    -------
    particles : np.ndarray
        Sampled particle positions.
        - 1D: shape (num_particles,)
        - nD: shape (num_particles, dimension)

    Examples
    --------
    >>> density_1d = np.exp(-((x - 0.5)**2) / 0.1)
    >>> particles = sample_particles_from_density(density_1d, 1000, ((0, 1),))
    """
    dimension = len(bounds)

    # Convert bounds to coordinate arrays (required by sample_from_density)
    coordinates = []
    for d in range(dimension):
        n_points = density.shape[d]
        coords = np.linspace(bounds[d][0], bounds[d][1], n_points)
        coordinates.append(coords)

    if implicit_domain is not None:
        # Domain-aware sampling (avoids obstacles)
        from mfgarchon.utils.numerical.particle import sample_from_density_with_domain

        particles = sample_from_density_with_domain(
            density=density,
            coordinates=coordinates,
            num_samples=num_particles,
            domain=implicit_domain,
        )
    else:
        # Standard sampling
        particles = sample_from_density(
            density=density,
            coordinates=coordinates,
            num_samples=num_particles,
        )

    # For 1D, return flat array (backward compatible)
    if dimension == 1 and particles.ndim > 1:
        return particles.ravel()

    return particles


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing unified fp_particle_density functions...")

    # Test 1: Brownian increments (1D and 2D)
    dW_1d = generate_brownian_increment(1000, 1, 0.01, 0.1)
    assert dW_1d.shape == (1000,), f"Expected (1000,), got {dW_1d.shape}"
    dW_2d = generate_brownian_increment(1000, 2, 0.01, 0.1)
    assert dW_2d.shape == (1000, 2), f"Expected (1000, 2), got {dW_2d.shape}"
    print("  generate_brownian_increment: OK (1D and 2D)")

    # Test 2: Density normalization (1D and 2D)
    density_1d = np.random.rand(50)
    normalized_1d = normalize_density(density_1d, 0.02)
    mass_1d = np.sum(normalized_1d) * 0.02
    assert np.abs(mass_1d - 1.0) < 1e-10, f"1D mass: {mass_1d}"

    density_2d = np.random.rand(20, 20)
    normalized_2d = normalize_density(density_2d, [0.05, 0.05])
    mass_2d = np.sum(normalized_2d) * 0.05 * 0.05
    assert np.abs(mass_2d - 1.0) < 1e-10, f"2D mass: {mass_2d}"
    print("  normalize_density: OK (1D and 2D)")

    # Test 3: Gradient computation (1D and 2D)
    x = np.linspace(0, 1, 50)
    field_1d = x**2
    grad_1d = compute_gradient(field_1d, 0.02)
    assert grad_1d.shape == (50,), f"Expected (50,), got {grad_1d.shape}"

    x2 = np.linspace(0, 1, 20)
    y2 = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x2, y2, indexing="ij")
    field_2d = X**2 + Y**2
    grad_2d = compute_gradient(field_2d, [0.05, 0.05])
    assert grad_2d.shape == (20, 20, 2), f"Expected (20, 20, 2), got {grad_2d.shape}"
    print("  compute_gradient: OK (1D and 2D)")

    # Test 4: KDE density estimation (1D)
    if SCIPY_AVAILABLE:
        particles_1d = np.random.normal(0.5, 0.1, 500)
        particles_1d = np.clip(particles_1d, 0, 1)
        density_est_1d = estimate_density_kde(particles_1d, (50,), ((0.0, 1.0),))
        assert density_est_1d.shape == (50,), f"Expected (50,), got {density_est_1d.shape}"
        assert np.all(density_est_1d >= 0)
        print("  estimate_density_kde (1D): OK")

    # Test 5: KDE density estimation (2D)
    particles_2d = np.random.normal(0.5, 0.15, (500, 2))
    particles_2d = np.clip(particles_2d, 0, 1)
    density_est_2d = estimate_density_kde(particles_2d, (20, 20), ((0.0, 1.0), (0.0, 1.0)))
    assert density_est_2d.shape == (20, 20), f"Expected (20, 20), got {density_est_2d.shape}"
    assert np.all(density_est_2d >= 0)
    print("  estimate_density_kde (2D): OK")

    # Test 6: Particle sampling (1D and 2D)
    # Create 1D density
    x_sample = np.linspace(0, 1, 50)
    density_for_sample = np.exp(-((x_sample - 0.5) ** 2) / 0.05)
    sampled_1d = sample_particles_from_density(density_for_sample, 200, ((0.0, 1.0),))
    assert sampled_1d.shape == (200,), f"Expected (200,), got {sampled_1d.shape}"

    # Create 2D density
    density_2d_sample = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.1)
    sampled_2d = sample_particles_from_density(density_2d_sample, 200, ((0.0, 1.0), (0.0, 1.0)))
    assert sampled_2d.shape == (200, 2), f"Expected (200, 2), got {sampled_2d.shape}"
    print("  sample_particles_from_density: OK (1D and 2D)")

    print("\nAll smoke tests passed!")
