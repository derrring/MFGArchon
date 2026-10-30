"""
Type protocols for PDE coefficient functions.

This module defines precise callable signatures for drift and diffusion
coefficients in PDEs, supporting both precomputed arrays and state-dependent
(nonlinear) formulations.

Usage:
    >>> from mfgarchon.types.pde_coefficients import DriftCallable, DiffusionCallable
    >>>
    >>> # Define state-dependent drift
    >>> def my_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
    ...     return -0.5 * x  # Linear drift
    >>>
    >>> # Type checking
    >>> assert isinstance(my_drift, DriftCallable)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class DriftCallable(Protocol):
    """
    Protocol for state-dependent drift field α(t, x, m).

    The drift field represents the deterministic part of the dynamics:
        dx_t = α(t, x_t, m_t) dt + σ dW_t

    Mathematical formulation:
        - FP equation: ∂m/∂t + ∇·(α m) = ∂ᵢ∂ⱼ(Dᵢⱼ m)   (Itô, the adjoint of the HJB's D:∇∇u; #2841)
        - HJB equation: -∂u/∂t + H(∇u, α) - D:∇∇u = 0

    Args:
        t: Current time (scalar in [0, T])
        x: Spatial coordinate(s)
           Shape conventions:
           - 1D: (Nx,) - grid points
           - 2D: (Nx, Ny, 2) - meshgrid with coordinates in last dimension
           - 3D: (Nx, Ny, Nz, 3) - meshgrid with coordinates in last dimension
           - Particles: (N_particles, d) - flattened format
        m: Current density field
           Shape conventions:
           - FDM: (*spatial_shape,) - grid density
           - Particles: (N_particles,) - pointwise density

    Returns:
        Drift vector field α(t,x,m):
        - 1D: (Nx,) - scalar drift
        - 2D: (Nx, Ny, 2) - vector drift [α_x, α_y]
        - 3D: (Nx, Ny, Nz, 3) - vector drift [α_x, α_y, α_z]
        - Particles: (N_particles, d) - per-particle drift

    Examples:
        >>> # 1D: Linear drift
        >>> def linear_drift(t, x, m):
        ...     return -0.5 * x  # x: (Nx,), returns: (Nx,)

        >>> # 2D: Rotation field
        >>> def rotation_drift(t, x, m):
        ...     drift = np.zeros_like(x)  # (Nx, Ny, 2)
        ...     drift[..., 0] = -x[..., 1]  # α_x = -y
        ...     drift[..., 1] = x[..., 0]   # α_y = x
        ...     return drift

        >>> # State-dependent: Crowd avoidance
        >>> def crowd_avoidance(t, x, m):
        ...     grad_m = np.gradient(m)  # Density gradient
        ...     return -np.stack(grad_m, axis=-1)  # Move down gradient

    Phase 2 Implementation Notes:
        - Solvers should call this function at each time step
        - Vectorized evaluation: pass entire spatial grid at once
        - Pointwise evaluation: call for each grid point (slower but flexible)
        - State-dependence enables nonlinear PDEs (porous medium, etc.)
    """

    def __call__(
        self,
        t: float,
        x: NDArray[np.floating],
        m: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate drift at given state."""
        ...


@runtime_checkable
class DiffusionCallable(Protocol):
    """
    Protocol for state-dependent diffusion coefficient D(t, x, m).

    Convention (Issue #811, physics/PDE standard):
        PDE:  dm/dt + div(alpha m) = D Laplacian(m)    (D appears directly)
        SDE:  dX = alpha dt + sigma dW,  where sigma = sqrt(2D)

    The callable returns the PDE diffusion coefficient D = sigma^2/2,
    NOT the SDE volatility sigma. This is the physics convention where
    D appears directly in the PDE without extra factors.

    Three forms supported:
    1. Isotropic: D = sigma^2/2 (scalar, same in all directions)
    2. Anisotropic: D = diag(sigma_1^2/2, ...) (diagonal, per direction)
    3. Full tensor: D = (1/2) Sigma Sigma^T (general SPD matrix)

    Args:
        t: Current time (scalar)
        x: Spatial coordinate(s) - same format as DriftCallable
        m: Current density field - same format as DriftCallable

    Returns:
        Diffusion coefficient/tensor D(t,x,m):
        - Scalar: float (isotropic, same everywhere)
        - Spatially varying scalar: (*spatial_shape,) (isotropic, varies in space)
        - Diagonal: (*spatial_shape, d) (anisotropic, d diagonal entries)
        - Full tensor: (*spatial_shape, d, d) (general elliptic operator)

    Examples:
        >>> # Isotropic constant: D = 0.05 (i.e. sigma = sqrt(2*0.05) ~ 0.316)
        >>> def constant_diffusion(t, x, m):
        ...     return 0.05  # PDE coefficient D

        >>> # State-dependent isotropic
        >>> def density_diffusion(t, x, m):
        ...     return 0.05 * (1.0 + m)  # D increases with density

        >>> # Anisotropic (2D): different D per direction
        >>> def anisotropic_2d(t, x, m):
        ...     D = np.zeros((*m.shape, 2))
        ...     D[..., 0] = 0.05  # D_x
        ...     D[..., 1] = 0.25  # D_y (faster diffusion in y)
        ...     return D

        >>> # Full tensor (2D, with cross-diffusion)
        >>> def full_tensor_2d(t, x, m):
        ...     D = np.zeros((*m.shape, 2, 2))
        ...     D[..., 0, 0] = 0.05
        ...     D[..., 1, 1] = 0.25
        ...     D[..., 0, 1] = D[..., 1, 0] = 0.02  # Cross-diffusion
        ...     return D

    Phase 2 Implementation Notes:
        - Return type determines form (scalar/diagonal/full)
        - Solvers must handle degenerate cases (D → 0 requires special numerics)
        - Anisotropic/full tensors require tensor-aware discretizations
        - State-dependence: porous medium equation, fast diffusion, etc.
    """

    def __call__(
        self,
        t: float,
        x: NDArray[np.floating],
        m: NDArray[np.floating],
    ) -> NDArray[np.floating] | float:
        """Evaluate diffusion at given state."""
        ...


# Type aliases for clarity in solver signatures
DriftField = float | NDArray[np.floating] | DriftCallable | None  # None → 0 (no drift)
DiffusionField = float | NDArray[np.floating] | DiffusionCallable | None  # None → 0 (deterministic)
