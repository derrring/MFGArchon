"""
GPU-accelerated utility functions for particle-based MFG solvers.

This module provides backend-aware operations for particle methods,
enabling full GPU acceleration when backend is available with automatic
fallback to NumPy/scipy for CPU-only environments.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mfgarchon.backends.base_backend import BaseBackend


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


#: Why "dirichlet" is refused by the 1-D helpers (#2485, #2550): they return positions, so they can only clamp.
_DIRICHLET_REFUSAL = (
    "bc_type 'dirichlet' is refused. This function returns positions, so it cannot remove a particle at an absorbing "
    "wall, and a clamp to the wall keeps the mass that should leave through it (#2485, #2550). Read which particles "
    "leave with mfgarchon.geometry.boundary.corner.absorb_at_walls(particles[:, None], [(xmin, xmax)], tolerance), at "
    "your reader's band (ParticleApplicator's is ONWALL_TOL, 1e-10); the particle solver's 1-D GPU path refuses an "
    "absorbing BC (#1910)."
)


def apply_boundary_conditions_gpu(particles, xmin: float, xmax: float, bc_type: str, backend: "BaseBackend"):
    """
    Apply boundary conditions to particles on GPU.

    Supports periodic and no-flux (reflecting) boundaries; "dirichlet" is refused by name.
    All operations performed in parallel on GPU.

    The fold and the wrap are the owners' own per-axis rules, `corner.fold_axis` and `periodic.wrap_axis`, on the
    device (CONVENTIONS.md § *Particles at a wall*; #2830). The NumPy fallback below reads the owners, and
    `test_particle_wall_owners_2550.py` pins this helper against them.

    Parameters
    ----------
    particles : backend tensor
        Particle positions, shape (N,)
    xmin : float
        Lower boundary
    xmax : float
        Upper boundary
    bc_type : str
        Boundary condition type: 'periodic' or 'no_flux'
    backend : BaseBackend
        Backend providing tensor operations

    Returns
    -------
    backend tensor
        Particles with boundary conditions applied, shape (N,)
    """
    if bc_type == "dirichlet":
        raise ValueError(f"apply_boundary_conditions_gpu: {_DIRICHLET_REFUSAL}")

    xp = backend.array_module
    Lx = xmax - xmin
    # The owner's refusal of a non-finite position (`corner.walls_reached`, #2550), on the device, with the owner's
    # message: the check is one scalar read per call, and the 1-D GPU loop already reads its drift back every step.
    if not bool(xp.isfinite(particles).all()):
        from mfgarchon.geometry.boundary.corner import non_finite_refusal

        finite = np.asarray(backend.to_numpy(xp.isfinite(particles)), dtype=bool).reshape(-1)
        row = int(np.flatnonzero(~finite)[0])
        value = float(np.asarray(backend.to_numpy(particles)).reshape(-1)[row])
        raise non_finite_refusal(row, [value], int((~finite).sum()), finite.size)

    if bc_type == "periodic":
        # The owner's per-axis wrap, on the device (#2830); an axis with no span is left as it is, as the owner leaves it
        if Lx > 1e-14:
            from mfgarchon.geometry.boundary.periodic import wrap_axis

            particles = wrap_axis(particles, xmin, xmax, xp)

    elif bc_type == "no_flux":
        # The owner's per-axis fold, on the device (#2830)
        if Lx > 1e-14:
            from mfgarchon.geometry.boundary.corner import fold_axis

            particles = fold_axis(particles, xmin, xmax, xp)

    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")

    return particles


def apply_boundary_conditions_numpy(particles: np.ndarray, xmin: float, xmax: float, bc_type: str) -> np.ndarray:
    """
    CPU fallback for applying boundary conditions, through their owners (CONVENTIONS.md § *Particles at a wall*):
    'periodic' wraps by `periodic.wrap_positions` and 'no_flux' folds by `corner.reflect_positions`. "dirichlet" is
    refused by name, and so is a non-finite position, by the owner's reach (`corner.walls_reached`). ``particles`` is
    not written.

    Parameters
    ----------
    particles : np.ndarray
        Particle positions, shape (N,)
    xmin : float
        Lower boundary
    xmax : float
        Upper boundary
    bc_type : str
        Boundary condition type: 'periodic' or 'no_flux'

    Returns
    -------
    np.ndarray
        Particles with boundary conditions applied, the input's shape; a float input keeps its dtype
    """
    from mfgarchon.geometry.boundary.corner import reflect_positions, walls_reached
    from mfgarchon.geometry.boundary.periodic import wrap_positions

    if bc_type == "dirichlet":
        raise ValueError(f"apply_boundary_conditions_numpy: {_DIRICHLET_REFUSAL}")
    positions = np.asarray(particles)
    if positions.dtype.kind != "f":
        positions = positions.astype(np.float64)
    column = positions.reshape(-1, 1)
    walls_reached(column, [(xmin, xmax)], 0.0)  # refuses a non-finite position by name

    if bc_type == "periodic":
        result = wrap_positions(column, [(xmin, xmax)])
    elif bc_type == "no_flux":
        result = reflect_positions(column, [(xmin, xmax)])
    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")

    return result.reshape(positions.shape)


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
        Random seed. Drives a LOCAL generator, so a seeded call is reproducible without touching
        any global stream. `seed=None` draws from -- and therefore advances -- the caller's global
        stream, which is the compatibility contract for callers that seed `torch` or `np.random`
        themselves.

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
        # Unreachable today, and by construction rather than by configuration: the only caller,
        # `_solve_fp_system_gpu`, assigns into a preallocated array (`X_particles_gpu[0, :] = ...`),
        # and a JAX array raises `TypeError: JAX arrays are immutable` on that. So the GPU solve
        # path is torch-only. Seeded anyway -- a `seed` argument that is silently ignored is a lie
        # whether or not anyone reaches it.
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
