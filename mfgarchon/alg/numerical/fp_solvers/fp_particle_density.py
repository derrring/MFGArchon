"""
Dimension-agnostic density estimation for particle-based FP solvers (Issue #635).

This module provides unified functions that work for any dimension:
- Density normalization
- Gradient computation
- Brownian increment generation

All functions accept both 1D and nD inputs with consistent APIs.
Internal optimizations (e.g., scipy fast path for 1D) are implementation details.

Usage:
    from mfgarchon.alg.numerical.fp_solvers.fp_particle_density import (
        normalize_density,
        compute_gradient,
        generate_brownian_increment,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

# Issue #625: Migrated from tensor_calculus to operators/stencils
from mfgarchon.operators.stencils.finite_difference import gradient_nd

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
    volatility: float,
    rng: ModuleType | np.random.Generator = np.random,
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
    volatility : float
        SDE volatility σ (the diffusion is D = σ²/2)
    rng : module or np.random.Generator, optional
        Source of the normal draws. Defaults to the global `np.random` module, which is what an
        unseeded `FPParticleSolver` passes; a seeded one passes its private Generator (Issue #1838).

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
        return volatility * rng.normal(0, np.sqrt(dt), num_particles)

    return volatility * rng.normal(0, np.sqrt(dt), (num_particles, dimension))


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

    print("\nAll smoke tests passed!")
