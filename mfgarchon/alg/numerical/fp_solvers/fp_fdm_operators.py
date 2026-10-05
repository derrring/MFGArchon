"""Finite difference operators for FP equation discretization.

This module provides common utilities for the FDM discretization
of the Fokker-Planck equation.

Module structure per issue #388:
    fp_fdm_operators.py - Common utilities

Advection Scheme Files:
    fp_fdm_alg_divergence_centered.py - divergence_centered (conservative, oscillates)
    fp_fdm_alg_divergence_upwind.py   - divergence_upwind (conservative via telescoping)

Scheme Comparison:
    | Scheme             | PDE Form   | Spatial | Conservative | Stable |
    |--------------------|------------|---------|--------------|--------|
    | divergence_centered| div(v*m)   | Central | YES (flux)   | Pe<2   |
    | divergence_upwind  | div(v*m)   | Upwind  | YES (flux)   | Always |

Functions:
    is_boundary_point: Utility to check if a point is on the boundary

Note:
    - Boundary condition enforcement is in fp_fdm_bc.py
    - Advection term computation is in fp_fdm_advection.py
"""

from __future__ import annotations


def is_boundary_point(multi_idx: tuple[int, ...], shape: tuple[int, ...], ndim: int) -> bool:
    """Check if a grid point is on the boundary."""
    return any(multi_idx[d] == 0 or multi_idx[d] == shape[d] - 1 for d in range(ndim))
