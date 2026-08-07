"""
GPU-accelerated utility functions for particle-based MFG solvers.

This module provides backend-aware operations for particle methods,
enabling full GPU acceleration when backend is available with automatic
fallback to NumPy/scipy for CPU-only environments.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mfgarchon.backends.backend_protocol import BaseBackend


def interpolate_1d_gpu(x_query, x_grid, y_grid, backend: "BaseBackend"):
    """
    GPU-accelerated 1D linear interpolation.

    Interpolates values from a regular grid to query points using
    linear interpolation. All operations performed on GPU in parallel.

    Algorithm:
    1. Find bracketing indices via searchsorted (log N complexity per query)
    2. Gather bracketing values (parallel)
    3. Compute linear weights (parallel)
    4. Interpolate (parallel)

    Expected speedup: 5-10x vs scipy.interpolate.interp1d

    Parameters
    ----------
    x_query : backend tensor
        Query points where interpolation is desired, shape (M,)
        Must be GPU tensor from backend
    x_grid : backend tensor
        Grid points (must be sorted), shape (N,)
        Must be GPU tensor from backend
    y_grid : backend tensor
        Values at grid points, shape (N,)
        Must be GPU tensor from backend
    backend : BaseBackend
        Backend providing tensor operations

    Returns
    -------
    backend tensor
        Interpolated values at query points, shape (M,)

    Examples
    --------
    >>> from mfgarchon.backends.torch_backend import TorchBackend
    >>> backend = TorchBackend(device='mps')
    >>> x_grid = backend.from_numpy(np.linspace(0, 1, 100))
    >>> y_grid = backend.from_numpy(np.sin(x_grid_np * np.pi))
    >>> x_query = backend.from_numpy(np.array([0.25, 0.5, 0.75]))
    >>> y_interp = interpolate_1d_gpu(x_query, x_grid, y_grid, backend)

    Notes
    -----
    - Assumes x_grid is sorted (ascending)
    - Query points outside [x_grid[0], x_grid[-1]] are clamped
    - Uses linear interpolation (C0 continuous)
    - All operations are GPU-parallel
    """
    xp = backend.array_module

    # Find bracketing indices
    # searchsorted returns index where x_query would be inserted
    # We want indices such that x_grid[idx-1] <= x_query < x_grid[idx]
    indices = xp.searchsorted(x_grid, x_query)

    # Clamp to valid range [1, len(x_grid)-1]
    # For extrapolation: clamp indices to valid bracket range
    indices = xp.clip(indices, 1, len(x_grid) - 1)

    # For query points outside grid, extrapolate using edge values
    # Points below grid: use first interval slope
    # Points above grid: use last interval slope

    # Bracketing indices
    idx_lo = indices - 1
    idx_up = indices

    # Gather values (parallel indexing)
    x_lo = x_grid[idx_lo]
    x_up = x_grid[idx_up]
    y_lo = y_grid[idx_lo]
    y_up = y_grid[idx_up]

    # Linear interpolation weight
    # w = (x - x_lo) / (x_up - x_lo)
    # y = y_lo + w * (y_up - y_lo)
    denominator = x_up - x_lo + 1e-10  # Avoid division by zero
    weight = (x_query - x_lo) / denominator

    # Interpolated values (all parallel)
    y_interp = y_lo + weight * (y_up - y_lo)

    return y_interp


def interpolate_1d_numpy(x_query: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """
    CPU fallback for 1D linear interpolation using NumPy.

    Uses NumPy's searchsorted and array indexing for CPU-based interpolation.
    Faster than scipy.interpolate.interp1d for single-shot interpolation.

    Parameters
    ----------
    x_query : np.ndarray
        Query points, shape (M,)
    x_grid : np.ndarray
        Grid points (sorted), shape (N,)
    y_grid : np.ndarray
        Values at grid points, shape (N,)

    Returns
    -------
    np.ndarray
        Interpolated values, shape (M,)
    """
    # Same algorithm as GPU version, using NumPy
    indices = np.searchsorted(x_grid, x_query)
    indices = np.clip(indices, 1, len(x_grid) - 1)

    idx_lo = indices - 1
    idx_up = indices

    x_lo = x_grid[idx_lo]
    x_up = x_grid[idx_up]
    y_lo = y_grid[idx_lo]
    y_up = y_grid[idx_up]

    weight = (x_query - x_lo) / (x_up - x_lo + 1e-10)
    y_interp = y_lo + weight * (y_up - y_lo)

    return y_interp


def apply_boundary_conditions_gpu(particles, xmin: float, xmax: float, bc_type: str, backend: "BaseBackend"):
    """
    Apply boundary conditions to particles on GPU.

    Supports periodic, no-flux (reflecting), and Dirichlet boundaries.
    All operations performed in parallel on GPU.

    Parameters
    ----------
    particles : backend tensor
        Particle positions, shape (N,)
    xmin : float
        Lower boundary
    xmax : float
        Upper boundary
    bc_type : str
        Boundary condition type: 'periodic', 'no_flux', or 'dirichlet'
    backend : BaseBackend
        Backend providing tensor operations

    Returns
    -------
    backend tensor
        Particles with boundary conditions applied, shape (N,)
    """
    xp = backend.array_module
    Lx = xmax - xmin

    if bc_type == "periodic":
        # Wrap around: x → xmin + (x - xmin) mod Lx
        particles = xmin + ((particles - xmin) % Lx)

    elif bc_type == "no_flux":
        # Reflecting boundaries using modular "fold" reflection
        # This handles particles that travel multiple domain widths in one step
        # Algorithm: position bounces back and forth with period 2*Lx
        if Lx > 1e-14:
            shifted = particles - xmin
            period = 2 * Lx
            pos_in_period = shifted % period
            in_second_half = pos_in_period > Lx
            # Reflect back: positions in [Lx, 2*Lx] map to [Lx, 0]
            particles = xp.where(
                in_second_half,
                xmin + period - pos_in_period,
                xmin + pos_in_period,
            )

    elif bc_type == "dirichlet":
        # Absorbing: clamp to domain
        # Backend compatibility - JAX uses clamp, NumPy uses clip (Issue #543 acceptable)
        if hasattr(xp, "clip"):
            particles = xp.clip(particles, xmin, xmax)
        else:
            # JAX doesn't have clip, use clamp
            particles = xp.minimum(xp.maximum(particles, xmin), xmax)

    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")

    return particles


def apply_boundary_conditions_numpy(particles: np.ndarray, xmin: float, xmax: float, bc_type: str) -> np.ndarray:
    """
    CPU fallback for applying boundary conditions.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions, shape (N,)
    xmin : float
        Lower boundary
    xmax : float
        Upper boundary
    bc_type : str
        Boundary condition type

    Returns
    -------
    np.ndarray
        Particles with boundary conditions applied
    """
    Lx = xmax - xmin

    if bc_type == "periodic":
        particles = xmin + ((particles - xmin) % Lx)

    elif bc_type == "no_flux":
        # Reflecting boundaries using modular "fold" reflection
        # This handles particles that travel multiple domain widths in one step
        Lx = xmax - xmin
        if Lx > 1e-14:
            shifted = particles - xmin
            period = 2 * Lx
            pos_in_period = shifted % period
            in_second_half = pos_in_period > Lx
            pos_in_period[in_second_half] = period - pos_in_period[in_second_half]
            particles = xmin + pos_in_period

    elif bc_type == "dirichlet":
        particles = np.clip(particles, xmin, xmax)

    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")

    return particles


def sample_from_density_gpu(density, grid, N: int, backend: "BaseBackend", seed: int | None = None):
    """
    Sample N particles from a probability density on GPU.

    Uses inverse transform sampling for 1D densities.

    Parameters
    ----------
    density : backend tensor
        Probability density on grid, shape (Nx,)
        Must integrate to 1 (normalized)
    grid : backend tensor
        Grid points, shape (Nx,)
    N : int
        Number of particles to sample
    backend : BaseBackend
        Backend providing tensor operations
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    backend tensor
        Sampled particle positions, shape (N,)

    Notes
    -----
    Algorithm:
    1. Compute CDF via cumulative sum
    2. Sample uniform random numbers U ~ Uniform(0, 1)
    3. Find x such that CDF(x) = U (inverse transform)
    4. Use searchsorted for fast inversion
    """
    xp = backend.array_module
    is_torch_backend = backend.__class__.__name__ == "TorchBackend"

    # Ensure density is normalized
    dx = grid[1] - grid[0]
    total_mass = xp.sum(density) * dx
    density_norm = density / (total_mass + 1e-10)

    # Compute CDF via cumulative sum
    # CDF(x_i) = ∫[x_0, x_i] ρ(x) dx ≈ Σ ρ(x_j) Δx
    pdf = density_norm * dx

    # PyTorch cumsum requires dim argument, NumPy doesn't
    pdf_flat = pdf.flatten()
    if is_torch_backend:
        cdf = xp.cumsum(pdf_flat, dim=0)
    else:
        cdf = xp.cumsum(pdf_flat)

    # Prepend 0 to CDF for proper inversion
    # Ensure zeros tensor matches cdf device and dtype for PyTorch
    if is_torch_backend:
        zeros = xp.zeros(1, device=cdf.device, dtype=cdf.dtype)
    else:
        zeros = xp.zeros(1)
    cdf_with_zero = xp.concatenate([zeros, cdf])

    # Sample uniform random numbers.
    #
    # The seed drives a LOCAL generator on both paths. `torch.manual_seed` reseeds the
    # process-global stream, so one seeded solve moved every unrelated torch draw in the same
    # interpreter -- the defect #1838 removed on the numpy side, reintroduced on this one the
    # moment a seed reached here. `seed=None` keeps drawing from each library's global stream,
    # which is the compatibility contract a caller that seeds `np.random` or `torch` relies on.
    if is_torch_backend:
        generator = None
        if seed is not None:
            generator = xp.Generator(device=cdf.device)
            generator.manual_seed(int(seed))
        U = xp.rand(N, device=cdf.device, dtype=cdf.dtype, generator=generator)
    else:
        draw = np.random if seed is None else np.random.default_rng(seed)
        U = backend.from_numpy(draw.random(N))

    # Inverse transform: find x such that CDF(x) = U
    indices = xp.searchsorted(cdf_with_zero, U)

    # Clamp to valid range
    indices = xp.clip(indices, 1, len(grid))

    # Sample particles at grid points (simple version)
    # More sophisticated: linear interpolation between grid points
    particles = grid[indices - 1]

    return particles


def sample_from_density_numpy(density: np.ndarray, grid: np.ndarray, N: int, seed: int | None = None) -> np.ndarray:
    """
    CPU fallback for sampling from density.

    Parameters
    ----------
    density : np.ndarray
        Probability density, shape (Nx,)
    grid : np.ndarray
        Grid points, shape (Nx,)
    N : int
        Number of samples
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Sampled particles, shape (N,)
    """
    if seed is not None:
        np.random.seed(seed)

    dx = grid[1] - grid[0]
    total_mass = np.sum(density) * dx
    density_norm = density / (total_mass + 1e-10)

    pdf = density_norm * dx
    cdf = np.cumsum(pdf)
    cdf_with_zero = np.concatenate([[0], cdf])

    U = np.random.rand(N)
    indices = np.searchsorted(cdf_with_zero, U)
    indices = np.clip(indices, 1, len(grid))

    particles = grid[indices - 1]

    return particles
