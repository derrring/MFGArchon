"""Finite difference operators for FP equation discretization.

This module provides common utilities and re-exports for the FDM discretization
of the Fokker-Planck equation.

Module structure per issue #388:
    fp_fdm_operators.py - Common utilities and backward-compatible re-exports

Advection Scheme Files (2x2 classification):
    fp_fdm_alg_gradient_centered.py   - gradient_centered (NOT mass-conservative at no-flux walls)
    fp_fdm_alg_gradient_upwind.py     - gradient_upwind (NOT mass-conservative at no-flux walls, Issue #1075)
    fp_fdm_alg_divergence_centered.py - divergence_centered (conservative, oscillates)
    fp_fdm_alg_divergence_upwind.py   - divergence_upwind (conservative via telescoping)

Scheme Comparison:
    | Scheme             | PDE Form   | Spatial | Conservative | Stable |
    |--------------------|------------|---------|--------------|--------|
    | gradient_centered  | v·grad(m)  | Central | NO           | Pe<2   |
    | gradient_upwind    | v·grad(m)  | Upwind  | YES (rows)   | Always |
    | divergence_centered| div(v*m)   | Central | YES (flux)   | Pe<2   |
    | divergence_upwind  | div(v*m)   | Upwind  | YES (flux)   | Always |

Functions:
    is_boundary_point: Utility to check if a point is on the boundary
    add_interior_entries: Re-export of gradient_upwind for backward compatibility

Note:
    - Boundary condition enforcement is in fp_fdm_bc.py
    - Advection term computation is in fp_fdm_advection.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Re-export gradient_upwind scheme for backward compatibility
from .fp_fdm_alg_gradient_upwind import add_interior_entries_gradient_upwind

if TYPE_CHECKING:
    import numpy as np


def is_boundary_point(multi_idx: tuple[int, ...], shape: tuple[int, ...], ndim: int) -> bool:
    """Check if a grid point is on the boundary."""
    return any(multi_idx[d] == 0 or multi_idx[d] == shape[d] - 1 for d in range(ndim))


def add_interior_entries(
    row_indices: list[int],
    col_indices: list[int],
    data_values: list[float],
    flat_idx: int,
    multi_idx: tuple[int, ...],
    shape: tuple[int, ...],
    ndim: int,
    dt: float,
    sigma: float,
    coupling_coefficient: float,
    spacing: tuple[float, ...],
    u_flat: np.ndarray,
    grid: Any,
    boundary_conditions: Any,
) -> None:
    """
    Add matrix entries for interior grid point using gradient_upwind FDM.

    This is a backward-compatible wrapper that calls add_interior_entries_gradient_upwind.
    For explicit scheme selection, use:
    - fp_fdm_alg_gradient_centered: add_interior_entries_gradient_centered (NOT mass-conservative at no-flux walls)
    - fp_fdm_alg_gradient_upwind: add_interior_entries_gradient_upwind (NOT mass-conservative at no-flux walls, Issue #1075)
    - fp_fdm_alg_divergence_centered: add_interior_entries_divergence_centered (conservative, oscillates)
    - fp_fdm_alg_divergence_upwind: add_interior_entries_divergence_upwind (conservative via telescoping)

    See individual module docstrings for mathematical details.
    """
    add_interior_entries_gradient_upwind(
        row_indices,
        col_indices,
        data_values,
        flat_idx,
        multi_idx,
        shape,
        ndim,
        dt,
        sigma,
        coupling_coefficient,
        spacing,
        u_flat,
        grid,
        boundary_conditions,
    )
