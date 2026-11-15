"""
GPU-accelerated utility functions for particle-based MFG solvers.

This module provides backend-aware operations for particle methods,
enabling full GPU acceleration when backend is available with automatic
fallback to NumPy/scipy for CPU-only environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.backends.base_backend import BaseBackend

logger = get_logger(__name__)


def interpolate_1d_gpu(x_query, x_grid, y_grid, backend: BaseBackend):
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


def apply_boundary_conditions_gpu(particles, xmin: float, xmax: float, bc_type: str, backend: BaseBackend):
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


def sample_from_density_gpu(density, grid, N: int, backend: BaseBackend, seed: int | None = None):
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
