"""
Particle Interpolation Utilities for MFG Computations.

This module provides functions for converting between grid-based and particle-based
representations, a common need in hybrid MFG solvers and visualization.

Key Functions:
- interpolate_grid_to_particles: Grid → Particles
- interpolate_particles_to_grid: Particles → Grid
- estimate_kde_bandwidth: Automatic bandwidth selection

Use Cases:
- Hybrid particle-grid methods (FP particle + HJB grid)
- Visualization of particle simulations on regular grids
- Coupling different solver types
- Initial condition generation

Examples:
    >>> import numpy as np
    >>> from mfgarchon.utils.numerical import interpolate_grid_to_particles
    >>>
    >>> # Create grid values
    >>> x = np.linspace(0, 1, 51)
    >>> u = np.exp(-10 * (x - 0.5)**2)
    >>>
    >>> # Interpolate to particles
    >>> particles = np.random.uniform(0, 1, 100)
    >>> u_particles = interpolate_grid_to_particles(
    ...     u, grid_bounds=(0, 1), particle_positions=particles
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.stats import gaussian_kde

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

InterpolationMethod = Literal["linear", "cubic", "rbf", "kde", "nearest"]


def interpolate_grid_to_particles(
    grid_values: NDArray[np.floating],
    grid_bounds: tuple[float, float] | tuple[tuple[float, float], ...],
    particle_positions: NDArray[np.floating],
    method: InterpolationMethod = "linear",
    *,
    grid_axes: Sequence[NDArray[np.floating]] | None = None,
) -> NDArray[np.floating]:
    """
    Interpolate values from regular grid to particle positions.

    This function takes values defined on a regular grid and evaluates them at
    arbitrary particle positions using the specified interpolation method.

    Parameters
    ----------
    grid_values : ndarray
        Values on regular grid. Shape:
        - 1D: (Nx,)
        - 2D: (Nx, Ny)
        - 3D: (Nx, Ny, Nz)
    grid_bounds : tuple
        Grid bounds:
        - 1D: (xmin, xmax)
        - 2D: ((xmin, xmax), (ymin, ymax))
        - 3D: ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    particle_positions : ndarray
        Particle positions. Shape:
        - 1D: (N,) or (N, 1)
        - 2D: (N, 2)
        - 3D: (N, 3)
    method : {"linear", "cubic", "nearest"}
        Interpolation method:
        - "linear": Fast, C0 continuous
        - "cubic": accepted, and interpolated linearly: no cubic path exists
        - "nearest": Fastest, discontinuous
    grid_axes : sequence of ndarray, optional
        The grid's node coordinates per dimension. Without them the axes are uniform over ``grid_bounds``,
        ``np.linspace(lo, hi, n)``, which misplaces the values of a graded grid; pass a graded grid's own (#2756).

    Returns
    -------
    ndarray
        Interpolated values at particle positions, shape (N,)

    Raises
    ------
    ValueError
        If dimensions don't match or method is unsupported

    Examples
    --------
    1D interpolation:

    >>> x_grid = np.linspace(0, 1, 51)
    >>> u_grid = np.sin(2 * np.pi * x_grid)
    >>> particles = np.array([0.25, 0.5, 0.75])
    >>> u_particles = interpolate_grid_to_particles(
    ...     u_grid, grid_bounds=(0, 1), particle_positions=particles
    ... )

    2D interpolation:

    >>> x = y = np.linspace(0, 1, 51)
    >>> X, Y = np.meshgrid(x, y, indexing='ij')
    >>> u_grid = np.exp(-10 * ((X - 0.5)**2 + (Y - 0.5)**2))
    >>> particles = np.random.uniform(0, 1, (100, 2))
    >>> u_particles = interpolate_grid_to_particles(
    ...     u_grid, grid_bounds=((0, 1), (0, 1)), particle_positions=particles
    ... )
    """
    # Normalize inputs
    grid_values = np.asarray(grid_values)
    particle_positions = np.asarray(particle_positions)

    # Determine dimension
    ndim = grid_values.ndim

    if ndim == 1:
        # 1D case
        if not isinstance(grid_bounds[0], (tuple, list)):
            xmin, xmax = grid_bounds
        else:
            xmin, xmax = grid_bounds[0]

        x = np.linspace(xmin, xmax, len(grid_values)) if grid_axes is None else np.asarray(grid_axes[0])

        # Ensure particle_positions is 1D
        if particle_positions.ndim == 2 and particle_positions.shape[1] == 1:
            particle_positions = particle_positions.ravel()

        interp = RegularGridInterpolator(
            (x,), grid_values, method=method if method != "cubic" else "linear", bounds_error=False, fill_value=0.0
        )

        return interp(particle_positions.reshape(-1, 1)).ravel()

    elif ndim == 2:
        # 2D case
        (xmin, xmax), (ymin, ymax) = grid_bounds
        nx, ny = grid_values.shape
        x, y = (np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)) if grid_axes is None else grid_axes

        # scipy wants (linear, cubic, quintic) for method name
        scipy_method = "linear" if method in ["linear", "cubic"] else "nearest"

        interp = RegularGridInterpolator((x, y), grid_values, method=scipy_method, bounds_error=False, fill_value=0.0)

        return interp(particle_positions)

    elif ndim == 3:
        # 3D case
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = grid_bounds
        nx, ny, nz = grid_values.shape
        x, y, z = (
            (np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), np.linspace(zmin, zmax, nz))
            if grid_axes is None
            else grid_axes
        )

        interp = RegularGridInterpolator(
            (x, y, z),
            grid_values,
            method="linear" if method != "nearest" else "nearest",
            bounds_error=False,
            fill_value=0.0,
        )

        return interp(particle_positions)

    else:
        raise ValueError(f"Unsupported grid dimension: {ndim}. Only 1D, 2D, and 3D are supported.")


def interpolate_particles_to_grid(
    particle_values: NDArray[np.floating],
    particle_positions: NDArray[np.floating],
    grid_shape: tuple[int, ...],
    grid_bounds: tuple[float, float] | tuple[tuple[float, float], ...],
    method: Literal["rbf", "kde", "nearest"] = "rbf",
    *,
    grid_axes: Sequence[NDArray[np.floating]] | None = None,
    **kwargs,
) -> NDArray[np.floating]:
    """
    Interpolate particle values to regular grid.

    This function takes values at scattered particle positions and creates a
    regular grid representation using the specified interpolation method.

    Parameters
    ----------
    particle_values : ndarray
        Values at particle positions, shape (N,)
    particle_positions : ndarray
        Particle positions. Shape:
        - 1D: (N,) or (N, 1)
        - 2D: (N, 2)
        - 3D: (N, 3)
    grid_shape : tuple
        Shape of output grid:
        - 1D: (Nx,)
        - 2D: (Nx, Ny)
        - 3D: (Nx, Ny, Nz)
    grid_bounds : tuple
        Grid bounds (same format as interpolate_grid_to_particles)
    grid_axes : sequence of ndarray, optional
        The grid's node coordinates per dimension, as in ``interpolate_grid_to_particles``; without them the grid is
        uniform over ``grid_bounds``.
    method : {"rbf", "kde", "nearest"}
        Interpolation method:
        - "rbf": Radial Basis Functions (smooth, accurate, slower)
        - "kde": Kernel Density Estimation (for density fields)
        - "nearest": Nearest neighbor (fast, discontinuous)
    **kwargs
        Additional parameters:
        - kernel: RBF kernel type ("gaussian", "multiquadric", "thin_plate_spline")
        - epsilon: RBF shape parameter (default: auto)
        - bandwidth: KDE bandwidth (default: auto via Scott's rule)

    Returns
    -------
    ndarray
        Grid values with shape matching grid_shape

    Examples
    --------
    1D particle-to-grid:

    >>> particles = np.random.uniform(0, 1, 100)
    >>> values = np.exp(-10 * (particles - 0.5)**2)
    >>> u_grid = interpolate_particles_to_grid(
    ...     values, particles, grid_shape=(51,), grid_bounds=(0, 1)
    ... )

    2D with RBF:

    >>> particles = np.random.uniform(0, 1, (200, 2))
    >>> values = np.exp(-10 * np.sum((particles - 0.5)**2, axis=1))
    >>> u_grid = interpolate_particles_to_grid(
    ...     values,
    ...     particles,
    ...     grid_shape=(51, 51),
    ...     grid_bounds=((0, 1), (0, 1)),
    ...     method="rbf",
    ...     kernel="thin_plate_spline"
    ... )
    """
    particle_values = np.asarray(particle_values)
    particle_positions = np.asarray(particle_positions)

    # Determine dimension
    if particle_positions.ndim == 1 or (particle_positions.ndim == 2 and particle_positions.shape[1] == 1):
        ndim = 1
        particle_positions = particle_positions.reshape(-1, 1)
    else:
        ndim = particle_positions.shape[1]

    # Create grid points
    if ndim == 1:
        xmin, xmax = grid_bounds if not isinstance(grid_bounds[0], (tuple, list)) else grid_bounds[0]
        x = np.linspace(xmin, xmax, grid_shape[0]) if grid_axes is None else np.asarray(grid_axes[0])
        grid_points = x.reshape(-1, 1)

    elif ndim == 2:
        (xmin, xmax), (ymin, ymax) = grid_bounds
        x, y = (
            (np.linspace(xmin, xmax, grid_shape[0]), np.linspace(ymin, ymax, grid_shape[1]))
            if grid_axes is None
            else grid_axes
        )
        X, Y = np.meshgrid(x, y, indexing="ij")
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

    elif ndim == 3:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = grid_bounds
        x, y, z = (
            (
                np.linspace(xmin, xmax, grid_shape[0]),
                np.linspace(ymin, ymax, grid_shape[1]),
                np.linspace(zmin, zmax, grid_shape[2]),
            )
            if grid_axes is None
            else grid_axes
        )
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    else:
        raise ValueError(f"Unsupported dimension: {ndim}")

    # Interpolate based on method
    if method == "rbf":
        kernel = kwargs.get("kernel", "thin_plate_spline")
        epsilon = kwargs.get("epsilon")

        rbf = RBFInterpolator(particle_positions, particle_values, kernel=kernel, epsilon=epsilon)

        grid_values = rbf(grid_points)

    elif method == "kde":
        # KDE is specifically for density estimation
        bandwidth = kwargs.get("bandwidth")

        if bandwidth is None:
            bandwidth = estimate_kde_bandwidth(particle_positions)

        kde = gaussian_kde(particle_positions.T, bw_method=bandwidth)
        grid_values = kde(grid_points.T)

    elif method == "nearest":
        from scipy.spatial import cKDTree

        tree = cKDTree(particle_positions)
        _, indices = tree.query(grid_points)
        grid_values = particle_values[indices]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'rbf', 'kde', or 'nearest'.")

    return grid_values.reshape(grid_shape)


def estimate_kde_bandwidth(
    particle_positions: NDArray[np.floating], method: Literal["scott", "silverman"] = "scott"
) -> float:
    """
    Estimate optimal KDE bandwidth using standard rules.

    Parameters
    ----------
    particle_positions : ndarray
        Particle positions, shape (N, d)
    method : {"scott", "silverman"}
        Bandwidth selection rule:
        - "scott": n^(-1/(d+4))
        - "silverman": (n*(d+2)/4)^(-1/(d+4))

    Returns
    -------
    float
        Estimated bandwidth

    Examples
    --------
    >>> particles = np.random.uniform(0, 1, (100, 2))
    >>> bw = estimate_kde_bandwidth(particles)
    >>> print(f"Optimal bandwidth: {bw:.4f}")
    """
    n, d = particle_positions.shape

    if method == "scott":
        return n ** (-1.0 / (d + 4))
    elif method == "silverman":
        return (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'scott' or 'silverman'.")


def kde_factor(bandwidth: float | str, particle_positions: NDArray[np.floating]) -> float:
    """
    The factor a KDE ``bandwidth`` names, for a sample (CONVENTIONS.md § 9, #2556).

    A bandwidth is a factor of the sample's spread, as in `scipy.stats.gaussian_kde`: ``"scott"`` or ``"silverman"``
    name scipy's rules, which `estimate_kde_bandwidth` computes, and a positive finite number is the factor itself.
    Anything else is refused by name: an absolute width is not a bandwidth, and goes through a separately named
    keyword at the reader that takes one.

    Parameters
    ----------
    bandwidth : float or {"scott", "silverman"}
        The bandwidth.
    particle_positions : ndarray
        The sample, shape (N,) or (N, d).

    Returns
    -------
    float
        The factor.

    Examples
    --------
    >>> particles = np.random.default_rng(0).normal(size=(100, 2))
    >>> round(kde_factor("scott", particles), 4)  # 100 ** (-1/6)
    0.4642
    >>> kde_factor(0.5, particles)
    0.5
    """
    points = np.asarray(particle_positions, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    if isinstance(bandwidth, str):
        return estimate_kde_bandwidth(points, method=bandwidth)  # type: ignore[arg-type]
    if (
        isinstance(bandwidth, (bool, np.bool_))
        or not isinstance(bandwidth, (int, float, np.integer, np.floating))
        or not np.isfinite(bandwidth)
        or bandwidth <= 0
    ):
        raise ValueError(
            f"A KDE bandwidth is a positive factor of the sample's standard deviation, or 'scott' or 'silverman', "
            f"as in scipy.stats.gaussian_kde (CONVENTIONS.md § 9); got {bandwidth!r}."
        )
    return float(bandwidth)


def kde_sample_has_spread(particle_positions: NDArray[np.floating]) -> bool:
    """
    Whether a sample has spread in every direction, so that a KDE of it has a kernel (#2556).

    A conditioning bound read in the sample's own units, so that scaling an axis leaves it unchanged, as it leaves
    scipy's KDE: in every direction, the sample's variance in the axes' own units must exceed the floor the positions'
    rounding sets there by 1e-14 of its largest variance. The floor on an axis is the square of 1e-13 of the positions'
    magnitude there (450 to 900 units in the last place); in the correlation's units it is that floor over the axis's
    variance, and a direction takes each axis's share of it. So the correlation matrix less those floors on its diagonal
    must have its smallest eigenvalue above 1e-14 of the correlation matrix's largest. The diagonal of that condition,
    each axis's variance above its floor, is checked first only because the correlation exists for positive variances
    alone. The covariance (ddof 1) is taken about the first particle, which leaves it unchanged and puts an axis on
    which every particle has one position at 0: `np.cov` gives such an axis, in more than one dimension, the square of
    its mean's accumulated rounding, which passes the floor from about 1e4 particles. False also for fewer than two
    particles.

    So a sample on a line or a plane is refused: `eigvalsh` returns its correlation's smallest eigenvalue as rounding,
    and when the line is short and far from the origin, as the positions' own rounding. A uniform sample on
    [0, 1e-7] x [0, 1], or 1e6 + 1e-5 N(0, I), is a spread, as scipy reads it, and so is an axis near its floor beside a
    thin direction that does not involve it. The bound's cost: a genuine sample is refused, though scipy builds a kernel
    for it, when its variance in some direction, in the axes' own units, does not exceed the rounding floor there by
    1e-14 of its largest: a spread in some direction below about 1e-7 of another, or within that floor (on an axis,
    1e-13 of its magnitude).

    Parameters
    ----------
    particle_positions : ndarray
        The sample, shape (N,) or (N, d). Non-finite positions are refused by name.
    """
    points = _sample(particle_positions)
    if len(points) < 2:
        return False
    covariance = _spread_covariance(points)
    variances = np.diag(covariance)
    floors = _rounding_variances(points)
    if not np.all(variances > floors):
        return False
    scale = np.sqrt(variances)
    correlation = covariance / np.outer(scale, scale)
    largest = np.linalg.eigvalsh(correlation)[-1]
    return bool(np.linalg.eigvalsh(correlation - np.diag(floors / variances))[0] > 1e-14 * largest)


def _sample(particle_positions: NDArray[np.floating]) -> NDArray[np.floating]:
    """The positions as a float array of shape (N, d), refusing a non-finite one by name."""
    points = np.asarray(particle_positions, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    if not np.all(np.isfinite(points)):
        raise ValueError(
            f"A KDE sample must have finite positions; {int(np.sum(~np.isfinite(points).all(axis=1)))} of "
            f"{len(points)} particle(s) have a NaN or infinite coordinate (#2556)."
        )
    return points


def _spread_covariance(points: NDArray[np.floating]) -> NDArray[np.floating]:
    """The sample covariance (ddof 1) taken about the first particle, which leaves it unchanged and puts an axis on
    which every particle has one position at 0, where `np.cov` gives it the square of its mean's accumulated
    rounding."""
    return np.atleast_2d(np.cov(points - points[0], rowvar=False))


def _rounding_variances(points: NDArray[np.floating]) -> NDArray[np.floating]:
    """Per axis, the floor below which a variance is taken for the positions' rounding: the square of 1e-13 of their
    magnitude there, about 450 units in the last place."""
    return (1e-13 * np.abs(np.mean(points, axis=0))) ** 2


def kde_kernel_covariance(particle_positions: NDArray[np.floating], bandwidth: float | str) -> NDArray[np.floating]:
    """
    The covariance of the Gaussian kernel a KDE of ``particle_positions`` uses, for ``bandwidth`` (CONVENTIONS.md § 9).

    It is ``kde_factor(bandwidth, particle_positions) ** 2`` times the sample covariance (ddof 1), the kernel
    `scipy.stats.gaussian_kde` builds, full in every dimension. A sample with no spread in some direction (all
    particles at one point, or on a line or a plane, or fewer than two: `kde_sample_has_spread`) has no kernel, and is
    refused by name.

    Parameters
    ----------
    particle_positions : ndarray
        The sample, shape (N,) or (N, d).
    bandwidth : float or {"scott", "silverman"}
        The bandwidth, as `kde_factor` reads it.

    Returns
    -------
    ndarray
        The kernel covariance, shape (d, d).
    """
    points = _sample(particle_positions)
    factor = kde_factor(bandwidth, points)
    n, d = points.shape
    if not kde_sample_has_spread(points):
        raise ValueError(
            f"A KDE of {n} particle(s) in {d}-D needs a sample with spread in every direction; this one has none at "
            "the conditioning bound `kde_sample_has_spread` states (one point, a line or a plane, fewer than two "
            "particles, or a spread in some direction below about 1e-7 of another in the axes' own units, or within "
            "the positions' rounding floor there), so it has no kernel (#2556)."
        )
    return factor**2 * np.atleast_2d(np.cov(points, rowvar=False))


def kde_kernel_widths(particle_positions: NDArray[np.floating], bandwidth: float | str) -> NDArray[np.floating]:
    """
    The per-axis widths of the kernel a KDE of ``particle_positions`` uses, for ``bandwidth`` (CONVENTIONS.md § 9).

    They are the square roots of `kde_kernel_covariance`'s diagonal, the factor times the sample's standard deviation
    (ddof 1) on each axis. Unlike the covariance they exist for a sample with no spread in some direction (on a line or
    a plane), whose width on that axis is zero to rounding, which is what a reader of one isotropic absolute width needs
    (#2556); they exist wherever the covariance does. A sample with no spread on any axis (one point, or fewer than two
    particles) has no width, and is refused by name.

    Parameters
    ----------
    particle_positions : ndarray
        The sample, shape (N,) or (N, d).
    bandwidth : float or {"scott", "silverman"}
        The bandwidth, as `kde_factor` reads it.

    Returns
    -------
    ndarray
        The widths, shape (d,).
    """
    points = _sample(particle_positions)
    if len(points) < 2 or not np.any(np.diag(_spread_covariance(points)) > _rounding_variances(points)):
        raise ValueError(
            f"A KDE of {len(points)} particle(s) has no kernel width: the sample has no spread on any axis (one point, "
            "or fewer than two particles) (#2556)."
        )
    return kde_factor(bandwidth, points) * np.sqrt(np.diag(np.atleast_2d(np.cov(points, rowvar=False))))


__all__ = [
    "estimate_kde_bandwidth",
    "kde_factor",
    "kde_kernel_covariance",
    "kde_kernel_widths",
    "kde_sample_has_spread",
    "interpolate_grid_to_particles",
    "interpolate_particles_to_grid",
]


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for particle interpolation utilities."""
    print("Testing particle interpolation utilities...")

    # Test 1D grid-to-particles
    x_grid = np.linspace(0, 1, 51)
    u_grid_1d = np.sin(2 * np.pi * x_grid)
    particles_1d = np.array([0.25, 0.5, 0.75])
    u_particles_1d = interpolate_grid_to_particles(u_grid_1d, grid_bounds=(0, 1), particle_positions=particles_1d)
    assert u_particles_1d.shape == (3,)
    assert np.allclose(u_particles_1d[1], 0.0, atol=1e-10)  # sin(π) ≈ 0
    print("✓ 1D grid-to-particles works")

    # Test 2D grid-to-particles
    x = y = np.linspace(0, 1, 51)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u_grid_2d = np.exp(-10 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
    particles_2d = np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]])
    u_particles_2d = interpolate_grid_to_particles(
        u_grid_2d, grid_bounds=((0, 1), (0, 1)), particle_positions=particles_2d, method="linear"
    )
    assert u_particles_2d.shape == (3,)
    assert u_particles_2d[0] > 0.9  # Center should be close to 1
    print("✓ 2D grid-to-particles works")

    # Test 3D grid-to-particles
    x = y = z = np.linspace(0, 1, 21)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    u_grid_3d = np.exp(-5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2))
    particles_3d = np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
    u_particles_3d = interpolate_grid_to_particles(
        u_grid_3d, grid_bounds=((0, 1), (0, 1), (0, 1)), particle_positions=particles_3d
    )
    assert u_particles_3d.shape == (2,)
    assert u_particles_3d[0] > 0.9  # Center should be close to 1
    print("✓ 3D grid-to-particles works")

    # Test 1D particles-to-grid (RBF)
    np.random.seed(42)
    particles_1d_scatter = np.random.uniform(0, 1, 50)
    values_1d = np.exp(-10 * (particles_1d_scatter - 0.5) ** 2)
    grid_1d_rbf = interpolate_particles_to_grid(
        values_1d, particles_1d_scatter, grid_shape=(51,), grid_bounds=(0, 1), method="rbf"
    )
    assert grid_1d_rbf.shape == (51,)
    assert grid_1d_rbf.max() > 0.5  # Should capture peak
    print("✓ 1D particles-to-grid (RBF) works")

    # Test 2D particles-to-grid (RBF)
    particles_2d_scatter = np.random.uniform(0, 1, (100, 2))
    values_2d = np.exp(-10 * np.sum((particles_2d_scatter - 0.5) ** 2, axis=1))
    grid_2d_rbf = interpolate_particles_to_grid(
        values_2d, particles_2d_scatter, grid_shape=(31, 31), grid_bounds=((0, 1), (0, 1)), method="rbf"
    )
    assert grid_2d_rbf.shape == (31, 31)
    assert grid_2d_rbf.max() > 0.5  # Should capture peak near center
    print("✓ 2D particles-to-grid (RBF) works")

    # Test KDE method
    particles_2d_kde = np.random.uniform(0.4, 0.6, (50, 2))  # Clustered particles
    values_2d_kde = np.ones(50)  # Uniform values
    grid_2d_kde = interpolate_particles_to_grid(
        values_2d_kde, particles_2d_kde, grid_shape=(31, 31), grid_bounds=((0, 1), (0, 1)), method="kde"
    )
    assert grid_2d_kde.shape == (31, 31)
    assert grid_2d_kde.max() > 0  # Should show density concentration
    print("✓ Particles-to-grid (KDE) works")

    # Test nearest neighbor method
    grid_2d_nearest = interpolate_particles_to_grid(
        values_2d, particles_2d_scatter, grid_shape=(31, 31), grid_bounds=((0, 1), (0, 1)), method="nearest"
    )
    assert grid_2d_nearest.shape == (31, 31)
    print("✓ Particles-to-grid (nearest) works")

    # Test bandwidth estimation
    bw_scott = estimate_kde_bandwidth(particles_2d_scatter, method="scott")
    bw_silverman = estimate_kde_bandwidth(particles_2d_scatter, method="silverman")
    assert 0 < bw_scott < 1
    assert 0 < bw_silverman < 1
    print("✓ KDE bandwidth estimation works")

    # Test different interpolation methods for grid-to-particles
    u_linear = interpolate_grid_to_particles(u_grid_2d, ((0, 1), (0, 1)), particles_2d, method="linear")
    u_nearest = interpolate_grid_to_particles(u_grid_2d, ((0, 1), (0, 1)), particles_2d, method="nearest")
    assert u_linear.shape == u_nearest.shape == (3,)
    print("✓ Different interpolation methods work")

    print("\nAll smoke tests passed!")
