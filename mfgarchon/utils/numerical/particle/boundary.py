"""
Particle Boundary Condition Utilities (1D, GPU/CPU).

This module provides 1D boundary condition functions for particle-based
MFG solvers. Supports both GPU and CPU backends.

For n-D position reflection/wrapping, use mfgarchon.geometry.boundary.corner instead:
    from mfgarchon.geometry.boundary.corner import reflect_positions, wrap_positions

Key Functions:
- apply_boundary_conditions_gpu: GPU-accelerated BC application (1D)
- apply_boundary_conditions_numpy: CPU fallback (1D)

Boundary Types:
- periodic: Wrap particles around domain boundaries
- no_flux: Reflect particles at boundaries (mass conserving)
- dirichlet: Absorbing boundaries (clamp to domain)

Examples:
    >>> import numpy as np
    >>> from mfgarchon.utils.numerical.particle import apply_boundary_conditions_numpy
    >>>
    >>> particles = np.array([-0.5, 0.5, 1.5, 2.5])
    >>> particles = apply_boundary_conditions_numpy(particles, 0.0, 2.0, "periodic")
    >>> print(particles)  # [1.5, 0.5, 1.5, 0.5]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.backends.base_backend import BaseBackend


def apply_boundary_conditions_gpu(
    particles,
    xmin: float,
    xmax: float,
    bc_type: str,
    backend: BaseBackend,
):
    """
    Apply boundary conditions to particles on GPU (1D).

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

    Notes
    -----
    - periodic: Particles wrap around the domain
    - no_flux: Particles reflect off boundaries (mass conserving)
    - dirichlet: Particles are clamped to domain (absorbing)

    For n-D positions, use mfgarchon.geometry.boundary.corner.reflect_positions instead.
    """
    xp = backend.array_module
    Lx = xmax - xmin

    if bc_type == "periodic":
        # Wrap around: x -> xmin + (x - xmin) mod Lx
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
        if hasattr(xp, "clip"):
            particles = xp.clip(particles, xmin, xmax)
        else:
            # JAX doesn't have clip, use clamp
            particles = xp.minimum(xp.maximum(particles, xmin), xmax)

    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")

    return particles


def apply_boundary_conditions_numpy(
    particles: NDArray[np.floating],
    xmin: float,
    xmax: float,
    bc_type: str,
) -> NDArray[np.floating]:
    """
    CPU fallback for applying boundary conditions (1D).

    Parameters
    ----------
    particles : np.ndarray
        Particle positions, shape (N,)
    xmin : float
        Lower boundary
    xmax : float
        Upper boundary
    bc_type : str
        Boundary condition type: 'periodic', 'no_flux', or 'dirichlet'

    Returns
    -------
    np.ndarray
        Particles with boundary conditions applied

    Examples
    --------
    >>> particles = np.array([-0.5, 0.5, 1.5, 2.5])
    >>> result = apply_boundary_conditions_numpy(particles, 0.0, 2.0, "periodic")
    >>> # particles wrap around: -0.5 -> 1.5, 2.5 -> 0.5

    Note
    ----
    For n-D positions, use mfgarchon.geometry.boundary.corner.reflect_positions instead.
    """
    # Make a copy to avoid modifying input
    particles = particles.copy()
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


__all__ = [
    "apply_boundary_conditions_gpu",
    "apply_boundary_conditions_numpy",
]


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    """Quick smoke test for 1D particle boundary utilities."""
    print("Testing 1D particle boundary utilities...")

    # Test periodic BC
    particles = np.array([-0.5, 0.5, 1.5, 2.5])
    result = apply_boundary_conditions_numpy(particles.copy(), 0.0, 2.0, "periodic")
    assert np.allclose(result, [1.5, 0.5, 1.5, 0.5])
    print("  Periodic BC: passed")

    # Test no-flux (reflecting) BC
    particles = np.array([-0.5, 0.5, 1.5, 2.5])
    result = apply_boundary_conditions_numpy(particles.copy(), 0.0, 2.0, "no_flux")
    assert np.allclose(result, [0.5, 0.5, 1.5, 1.5])  # -0.5 -> 0.5, 2.5 -> 1.5
    print("  No-flux BC: passed")

    # Test Dirichlet (absorbing) BC
    particles = np.array([-0.5, 0.5, 1.5, 2.5])
    result = apply_boundary_conditions_numpy(particles.copy(), 0.0, 2.0, "dirichlet")
    assert np.allclose(result, [0.0, 0.5, 1.5, 2.0])  # clamped
    print("  Dirichlet BC: passed")

    # Test that original is not modified
    original = np.array([2.5])
    result = apply_boundary_conditions_numpy(original, 0.0, 2.0, "no_flux")
    assert original[0] == 2.5, "Original should not be modified"
    print("  No mutation: passed")

    # Test no-flux with large displacements (multiple domain widths)
    particles = np.array([-5.0, 7.0, -3.0, 5.0])
    result = apply_boundary_conditions_numpy(particles.copy(), 0.0, 2.0, "no_flux")
    expected = np.array([1.0, 1.0, 1.0, 1.0])  # All fold back to 1.0
    assert np.allclose(result, expected), f"Large displacement test failed: {result} vs {expected}"
    print("  No-flux large displacement: passed")

    print("\nAll 1D smoke tests passed!")
    print("\nNote: For n-D positions, use mfgarchon.geometry.boundary.corner.reflect_positions")
