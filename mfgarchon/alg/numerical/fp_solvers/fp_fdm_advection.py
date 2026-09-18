"""Advection term discretization for FP FDM solver.

This module provides advection term computation using upwind schemes
for the Fokker-Planck finite difference solver.

Module structure per issue #388:
    fp_fdm_advection.py - How transport is discretized (upwinding, centered schemes)

Functions:
    compute_advection_term_nd: Compute div(alpha * m) using upwind scheme
    compute_advection_from_drift_nd: Compute div(alpha * m) with drift provided directly

Mathematical Background:
    Advection term: div(alpha * m)

    Two modes:
    1. MFG coupled: alpha = -coupling_coefficient * grad(U) (derived from HJB)
    2. Standalone FP: alpha provided directly by user

    Uses upwind scheme for stability in advection-dominated problems:
    - Positive velocity: backward difference (upwind from left)
    - Negative velocity: forward difference (upwind from right)

Issue #597 Milestone 3 Integration:
    As of v0.18.0, this module uses AdvectionOperator internally for explicit
    advection term evaluation. The legacy hand-rolled upwind helper it replaced
    was removed in v0.22.0 (#2343).

    Note: For implicit solvers, sparse matrix construction still uses the manual
    velocity-based upwind logic (fp_fdm_alg_*.py files). This hybrid approach
    is correct: linear velocity-based Jacobian (implicit LHS) with Godunov
    residuals (explicit RHS) is a standard Defect Correction strategy.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_advection_term_nd(
    M: np.ndarray,
    U: np.ndarray,
    coupling_coefficient: float,
    spacing: tuple[float, ...],
    ndim: int,
    boundary_conditions: Any,
    scheme: str = "upwind",
    mass_conservative: bool = False,
) -> np.ndarray:
    """
    Compute advection term div(alpha * m) where alpha = -coupling_coefficient * grad(U).

    This is the MFG-coupled mode where drift is derived from HJB value function U
    via Pontryagin's maximum principle.

    Parameters
    ----------
    M : np.ndarray
        Density field
    U : np.ndarray
        Value function from HJB equation
    coupling_coefficient : float
        Coupling coefficient for drift term (λ in v = -λ∇U)
    spacing : tuple
        Grid spacing (dx, dy, ...)
    ndim : int
        Spatial dimension
    boundary_conditions : BoundaryConditions
        Boundary conditions specification
    scheme : str, optional
        Advection scheme: "upwind" (default) or "centered"
    mass_conservative : bool, optional
        If True (upwind only), use the discretely-conservative finite-volume divergence that
        zeroes the advective flux through no-flux walls (Issue #1184/#1428), so density driven
        against a wall by strong drift does not leak mass. Default False takes the node-based
        divergence upwinded by the sign of the velocity (#2309).

    Returns
    -------
    np.ndarray
        Advection term div(alpha * m), same shape as M

    Notes
    -----
    **Issue #597 Milestone 3**: As of v0.18.0, this function uses AdvectionOperator
    internally. The upwind scheme provides unconditional stability for
    advection-dominated problems.

    The operator computes the divergence form: ∇·(vm) using velocity-based
    upwinding (not Godunov). This is correct for explicit time-stepping and
    residual evaluation.
    """
    from mfgarchon.operators import AdvectionOperator

    # Compute drift from U: alpha_d = -coupling_coefficient * grad_U_d
    drift_per_dim = []
    for d in range(ndim):
        dx = spacing[d]
        grad_U_d = np.gradient(U, dx, axis=d)
        alpha_d = -coupling_coefficient * grad_U_d
        drift_per_dim.append(alpha_d)

    # Stack drift into velocity field format: (ndim, Nx, Ny, ...)
    velocity_field = np.stack(drift_per_dim, axis=0)

    # Create advection operator
    adv_op = AdvectionOperator(
        velocity_field=velocity_field,
        spacings=list(spacing),
        field_shape=M.shape,
        scheme=scheme,
        form="divergence",  # Conservative form: ∇·(vm)
        bc=boundary_conditions,
        mass_conservative=mass_conservative,  # Issue #1428: zero advective flux through no-flux walls
    )

    # Apply operator
    return adv_op(M)


def compute_advection_from_drift_nd(
    M: np.ndarray,
    drift: np.ndarray,
    spacing: tuple[float, ...],
    ndim: int,
    scheme: str = "upwind",
    bc: Any | None = None,
    mass_conservative: bool = False,
) -> np.ndarray:
    """
    Compute advection term div(alpha * m) with drift alpha provided directly.

    This is the standalone FP mode where user provides drift field directly,
    without going through HJB value function.

    Parameters
    ----------
    M : np.ndarray
        Density field with shape (N1, N2, ..., Nd)
    drift : np.ndarray
        Drift field. For 1D: shape (N,) scalar drift.
        For nD: shape (ndim, N1, N2, ..., Nd) vector drift.
    spacing : tuple[float, ...]
        Grid spacing (dx, dy, ...)
    ndim : int
        Spatial dimension
    scheme : str, optional
        Advection scheme: "upwind" (default) or "centered"
    bc : BoundaryConditions, optional
        Boundary conditions. If None, uses periodic.
    mass_conservative : bool, optional
        If True (upwind only), use the discretely-conservative finite-volume divergence
        that zeroes the advective flux through no-flux walls, so a density driven against
        a wall by strong drift does not leak mass (Issue #1184). Default False takes the
        node-based divergence upwinded by the sign of the velocity (#2309).

    Returns
    -------
    np.ndarray
        Advection term div(alpha * m), same shape as M

    Notes
    -----
    **Issue #597 Milestone 3**: As of v0.18.0, this function uses AdvectionOperator
    internally.

    This function directly uses the provided drift without any conversion.
    For MFG systems where drift comes from HJB via v = -λ∇U, use
    compute_advection_term_nd instead.
    """
    from mfgarchon.operators import AdvectionOperator

    # Parse drift into per-dimension arrays and convert to velocity field
    if ndim == 1:
        # 1D: drift is scalar field
        if drift.ndim == 1:
            velocity_field = np.expand_dims(drift, axis=0)  # (1, N)
        elif drift.ndim == 2 and drift.shape[0] == 1:
            velocity_field = drift
        else:
            velocity_field = np.expand_dims(drift.ravel(), axis=0)
    else:
        # nD: drift should be vector field (ndim, N1, N2, ...)
        if drift.ndim == ndim + 1 and drift.shape[0] == ndim:
            velocity_field = drift
        elif drift.ndim == ndim:
            # Scalar drift applied to first dimension only (simplified case)
            velocity_field = np.zeros((ndim, *M.shape))
            velocity_field[0] = drift
        else:
            raise ValueError(
                f"Drift shape {drift.shape} incompatible with {ndim}D grid. "
                f"Expected ({ndim}, ...) for vector drift or (...) for scalar drift."
            )

    # Create advection operator
    adv_op = AdvectionOperator(
        velocity_field=velocity_field,
        spacings=list(spacing),
        field_shape=M.shape,
        scheme=scheme,
        form="divergence",  # Conservative form: ∇·(vm)
        bc=bc,
        mass_conservative=mass_conservative,
    )

    # Apply operator
    return adv_op(M)
