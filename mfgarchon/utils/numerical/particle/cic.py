"""
Cloud-in-Cell (CIC) density estimation for particle methods.

This module implements the standard Particle-Mesh (PM) technique:
1. Deposit particles onto regular grid via CIC (Cloud-in-Cell)
2. Interpolate grid density to arbitrary query points

This is the "industrial standard" in computational physics [Hockney & Eastwood 1988],
providing guaranteed O(h²) accuracy with mass conservation.

Mathematical Background:
    CIC Deposition (2D):
        For particle at (x_p, y_p), distribute mass to 4 nearest grid nodes
        using bilinear weights: w = (1-fx)(1-fy), fx(1-fy), (1-fx)fy, fx·fy
        where (fx, fy) is the fractional position within the cell.

    Properties:
        - Mass conservation: Σ m_ij × ΔV = 1 (exact)
        - Smoothness: C⁰ continuous density field
        - Accuracy: O(Δx²) spatial error
        - Partition of Unity: Σ w = 1 for any particle position

Usage for GFDM:
    # Particles → Grid (CIC) → Collocation Points (interpolation)
    m_grid = cic_deposit_2d(particles, bounds, n_grid, periodic=True)
    m_coll = interpolate_to_collocation(m_grid, collocation_points, bounds)

References:
    - Hockney & Eastwood (1988), "Computer Simulation Using Particles", Ch. 5
    - Issue #721: CIC density estimation for particle methods

Created: 2026-02-02
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


def cic_deposit_2d(
    particles: NDArray[np.float64],
    bounds: NDArray[np.float64] | list[tuple[float, float]],
    n_grid: int | tuple[int, int],
    periodic: bool = True,
) -> NDArray[np.float64]:
    """
    Standard CIC deposition onto regular 2D grid.

    Parameters
    ----------
    particles : NDArray
        Particle positions, shape (N_p, 2).
    bounds : array-like
        Domain bounds [[xmin, xmax], [ymin, ymax]].
    n_grid : int or tuple
        Grid resolution. If int, use (n_grid, n_grid).
    periodic : bool, default=True
        If True, use periodic boundary conditions (wrap indices).

    Returns
    -------
    m_grid : NDArray
        Density on grid, shape (nx, ny). Grid is cell-centered.

    Notes
    -----
    The grid is cell-centered, meaning grid point (i, j) represents
    the density at position:
        x_i = xmin + (i + 0.5) × Δx
        y_j = ymin + (j + 0.5) × Δy

    This matches the convention used by scipy.interpolate for periodic data.
    """
    particles = np.atleast_2d(particles)
    N_p = len(particles)

    if isinstance(n_grid, int):
        nx, ny = n_grid, n_grid
    else:
        nx, ny = n_grid

    bounds = np.asarray(bounds)
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    Lx, Ly = xmax - xmin, ymax - ymin
    dx, dy = Lx / nx, Ly / ny

    m_grid = np.zeros((nx, ny))

    for x, y in particles:
        # Continuous grid coordinates (cell-centered: subtract 0.5)
        gx = (x - xmin) / dx - 0.5
        gy = (y - ymin) / dy - 0.5

        # Integer cell indices (lower-left corner)
        ix = int(np.floor(gx))
        iy = int(np.floor(gy))

        # Fractional position within cell [0, 1)
        fx = gx - ix
        fy = gy - iy

        # CIC bilinear weights
        w00 = (1 - fx) * (1 - fy)  # (ix, iy)
        w10 = fx * (1 - fy)  # (ix+1, iy)
        w01 = (1 - fx) * fy  # (ix, iy+1)
        w11 = fx * fy  # (ix+1, iy+1)

        # Grid indices with boundary handling
        if periodic:
            ix0, ix1 = ix % nx, (ix + 1) % nx
            iy0, iy1 = iy % ny, (iy + 1) % ny
        else:
            # Clamp to grid boundaries
            ix0 = max(0, min(ix, nx - 1))
            ix1 = max(0, min(ix + 1, nx - 1))
            iy0 = max(0, min(iy, ny - 1))
            iy1 = max(0, min(iy + 1, ny - 1))

        # Deposit mass
        m_grid[ix0, iy0] += w00
        m_grid[ix1, iy0] += w10
        m_grid[ix0, iy1] += w01
        m_grid[ix1, iy1] += w11

    # Convert particle counts to density
    cell_volume = dx * dy
    m_grid /= N_p * cell_volume

    return m_grid


def cic_deposit_nd(
    particles: NDArray[np.float64],
    bounds: NDArray[np.float64],
    n_grid: int | tuple[int, ...],
    periodic: bool = True,
) -> NDArray[np.float64]:
    """
    N-dimensional CIC deposition onto regular grid.

    Parameters
    ----------
    particles : NDArray
        Particle positions, shape (N_p, d).
    bounds : NDArray
        Domain bounds, shape (d, 2) as [[xmin, xmax], ...].
    n_grid : int or tuple
        Grid resolution per dimension.
    periodic : bool, default=True
        Use periodic boundary conditions.

    Returns
    -------
    m_grid : NDArray
        Density on grid, shape (n1, n2, ..., nd).
    """
    particles = np.atleast_2d(particles)
    N_p, d = particles.shape

    bounds = np.asarray(bounds)
    if isinstance(n_grid, int):
        n_grid = (n_grid,) * d
    n_grid = np.array(n_grid)

    # Grid spacing
    L = bounds[:, 1] - bounds[:, 0]
    dx = L / n_grid

    m_grid = np.zeros(tuple(n_grid))
    cell_volume = np.prod(dx)

    # Pre-compute 2^d corner offsets
    corner_offsets = np.array(np.meshgrid(*([[0, 1]] * d), indexing="ij")).reshape(d, -1).T

    for p in particles:
        # Continuous grid coordinates (cell-centered)
        g = (p - bounds[:, 0]) / dx - 0.5

        # Integer indices and fractional parts
        idx = np.floor(g).astype(int)
        frac = g - idx

        # Loop over 2^d corners
        for offset in corner_offsets:
            corner_idx = idx + offset

            # Weight = product of 1D weights
            weight = 1.0
            for dim in range(d):
                if offset[dim] == 0:
                    weight *= 1 - frac[dim]
                else:
                    weight *= frac[dim]

            # Apply boundary conditions
            if periodic:
                corner_idx = corner_idx % n_grid
            else:
                corner_idx = np.clip(corner_idx, 0, n_grid - 1)

            m_grid[tuple(corner_idx)] += weight

    m_grid /= N_p * cell_volume
    return m_grid


def interpolate_grid_to_points(
    m_grid: NDArray[np.float64],
    query_points: NDArray[np.float64],
    bounds: NDArray[np.float64],
    periodic: bool = True,
    method: str = "linear",
) -> NDArray[np.float64]:
    """
    Interpolate density from regular grid to arbitrary query points.

    Parameters
    ----------
    m_grid : NDArray
        Density on regular grid, shape (n1, n2, ...).
    query_points : NDArray
        Points to interpolate to, shape (N_query, d).
    bounds : NDArray
        Domain bounds, shape (d, 2).
    periodic : bool, default=True
        Use periodic boundary handling (wrap grid edges).
    method : str, default="linear"
        Interpolation method: "linear" (bilinear) or "nearest".

    Returns
    -------
    m_query : NDArray
        Interpolated density at query points, shape (N_query,).

    Notes
    -----
    For periodic domains, the grid is padded with wrapped values to ensure
    smooth interpolation across boundaries.
    """
    bounds = np.asarray(bounds)
    d = len(bounds)
    n_grid = np.array(m_grid.shape)

    # Grid cell centers
    axes = []
    for dim in range(d):
        L = bounds[dim, 1] - bounds[dim, 0]
        dx = L / n_grid[dim]
        # Cell centers
        x = bounds[dim, 0] + dx * (np.arange(n_grid[dim]) + 0.5)
        axes.append(x)

    if periodic:
        # Pad grid with wrapped values for smooth periodic interpolation
        pad_width = [(1, 1)] * d
        m_padded = np.pad(m_grid, pad_width, mode="wrap")

        # Extend axes
        axes_padded = []
        for dim in range(d):
            dx = (bounds[dim, 1] - bounds[dim, 0]) / n_grid[dim]
            x_padded = np.concatenate([[axes[dim][0] - dx], axes[dim], [axes[dim][-1] + dx]])
            axes_padded.append(x_padded)

        interp = RegularGridInterpolator(
            tuple(axes_padded), m_padded, method=method, bounds_error=False, fill_value=None
        )
    else:
        interp = RegularGridInterpolator(tuple(axes), m_grid, method=method, bounds_error=False, fill_value=0.0)

    return interp(query_points)


class GridCIC:
    """
    Grid-based CIC density estimator for particle methods.

    This class wraps the CIC deposition and interpolation functions,
    providing a convenient interface for GFDM solvers.

    Parameters
    ----------
    bounds : array-like
        Domain bounds [[xmin, xmax], [ymin, ymax], ...].
    n_grid : int or tuple
        Grid resolution per dimension.
    periodic : bool, default=True
        Use periodic boundary conditions.

    Examples
    --------
    >>> bounds = np.array([[0, 2], [0, 2]])
    >>> cic = GridCIC(bounds, n_grid=64, periodic=True)
    >>>
    >>> # In FP solver loop:
    >>> particles = propagate_sde(particles, drift, sigma, dt)
    >>> m_coll = cic.estimate_density(particles, collocation_points)
    """

    def __init__(
        self,
        bounds: NDArray[np.float64],
        n_grid: int | tuple[int, ...],
        periodic: bool = True,
    ):
        self.bounds = np.asarray(bounds)
        self.d = len(self.bounds)
        self.n_grid = n_grid if isinstance(n_grid, tuple) else (n_grid,) * self.d
        self.periodic = periodic

        # Grid spacing
        L = self.bounds[:, 1] - self.bounds[:, 0]
        self.dx = L / np.array(self.n_grid)

        logger.debug(f"GridCIC: d={self.d}, n_grid={self.n_grid}, dx={self.dx}, periodic={self.periodic}")

    def deposit(self, particles: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Deposit particles onto regular grid.

        Parameters
        ----------
        particles : NDArray
            Particle positions, shape (N_p, d).

        Returns
        -------
        m_grid : NDArray
            Density on grid.
        """
        if self.d == 2:
            return cic_deposit_2d(particles, self.bounds, self.n_grid, self.periodic)
        else:
            return cic_deposit_nd(particles, self.bounds, self.n_grid, self.periodic)

    def interpolate(
        self,
        m_grid: NDArray[np.float64],
        query_points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Interpolate grid density to query points.
        """
        return interpolate_grid_to_points(m_grid, query_points, self.bounds, periodic=self.periodic, method="linear")

    def estimate_density(
        self,
        particles: NDArray[np.float64],
        query_points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Full pipeline: deposit + interpolate.

        Parameters
        ----------
        particles : NDArray
            Particle positions, shape (N_p, d).
        query_points : NDArray
            Points to query density at, shape (N_query, d).

        Returns
        -------
        m : NDArray
            Density at query points, shape (N_query,).
        """
        m_grid = self.deposit(particles)
        return self.interpolate(m_grid, query_points)


# =============================================================================
# SMOKE TEST
# =============================================================================
