"""
Discrete adjoint verification utilities.

This module provides tools for verifying that HJB and FP discrete operators
satisfy the adjoint relationship A_FP = A_HJB^T.

Key functions:
- verify_discrete_adjoint: Check matrix transpose relationship
- verify_duality: Check ⟨m, L_hjb U⟩ = ⟨L_fp* m, U⟩
- compute_adjoint_error: Detailed error breakdown

References:
-----------
- Issue #704: Unified adjoint module
- Issue #625 (state-dependent BC coupling)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass
class AdjointVerificationResult:
    """Result of discrete adjoint verification."""

    is_adjoint: bool
    """Whether A_fp ≈ A_hjb^T within tolerance."""

    frobenius_error: float
    """Frobenius norm of A_fp - A_hjb^T."""

    relative_error: float
    """Relative error: ||A_fp - A_hjb^T||_F / ||A_hjb||_F."""

    max_error: float
    """Maximum absolute entry-wise error."""

    max_error_location: tuple[int, int]
    """(row, col) of maximum error."""

    interior_error: float
    """Error from interior points only."""

    boundary_error: float
    """Error from boundary rows/columns only."""

    n_interior: int
    """Number of interior points."""

    n_boundary: int
    """Number of boundary points."""


@dataclass
class DualityVerificationResult:
    """Result of duality verification ⟨m, L_hjb U⟩ = ⟨L_fp* m, U⟩."""

    lhs: float
    """Left-hand side: ⟨m, L_hjb(U)⟩."""

    rhs: float
    """Right-hand side: ⟨L_fp*(m), U⟩."""

    absolute_error: float
    """Absolute error |lhs - rhs|."""

    relative_error: float
    """Relative error |lhs - rhs| / max(|lhs|, |rhs|)."""

    is_dual: bool
    """Whether duality holds within tolerance."""


def verify_discrete_adjoint(
    A_hjb: sparse.spmatrix | NDArray,
    A_fp: sparse.spmatrix | NDArray,
    rtol: float = 1e-10,
    boundary_indices: NDArray | None = None,
) -> AdjointVerificationResult:
    """
    Verify A_fp = A_hjb^T within tolerance.

    Args:
        A_hjb: HJB discrete operator matrix
        A_fp: FP discrete operator matrix
        rtol: Relative tolerance for adjoint check
        boundary_indices: Indices of boundary points (for error decomposition).
            If None, assumes first and last rows/cols are boundary (1D).

    Returns:
        AdjointVerificationResult with detailed error breakdown.

    Example:
        >>> from mfgarchon.alg.numerical.adjoint import verify_discrete_adjoint
        >>> result = verify_discrete_adjoint(A_hjb, A_fp)
        >>> if not result.is_adjoint:
        ...     print(f"Adjoint error: {result.relative_error:.2e}")
        ...     print(f"Boundary error: {result.boundary_error:.2e}")
    """
    # Convert to dense for analysis (sparse OK for large matrices)
    if sparse.issparse(A_hjb):
        A_hjb = A_hjb.toarray()
    if sparse.issparse(A_fp):
        A_fp = A_fp.toarray()

    A_hjb = np.asarray(A_hjb)
    A_fp = np.asarray(A_fp)

    n = A_hjb.shape[0]

    # Compute difference
    diff = A_fp - A_hjb.T

    # Overall error
    frobenius_error = np.linalg.norm(diff, "fro")
    hjb_norm = np.linalg.norm(A_hjb, "fro")
    relative_error = frobenius_error / hjb_norm if hjb_norm > 0 else frobenius_error

    # Maximum error location
    max_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    max_error = np.abs(diff[max_idx])

    # Boundary vs interior decomposition
    if boundary_indices is None:
        # Default: first and last are boundary (1D case)
        boundary_indices = np.array([0, n - 1])

    boundary_set = set(boundary_indices)
    interior_indices = np.array([i for i in range(n) if i not in boundary_set])

    # Interior error: rows and cols both interior
    if len(interior_indices) > 0:
        interior_mask = np.ix_(interior_indices, interior_indices)
        interior_diff = diff[interior_mask]
        interior_error = np.linalg.norm(interior_diff, "fro")
    else:
        interior_error = 0.0

    # Boundary error: at least one of row or col is boundary
    boundary_error = np.sqrt(max(0, frobenius_error**2 - interior_error**2))

    is_adjoint = relative_error < rtol

    return AdjointVerificationResult(
        is_adjoint=is_adjoint,
        frobenius_error=frobenius_error,
        relative_error=relative_error,
        max_error=max_error,
        max_error_location=max_idx,
        interior_error=interior_error,
        boundary_error=boundary_error,
        n_interior=len(interior_indices),
        n_boundary=len(boundary_indices),
    )


def verify_duality(
    m: NDArray,
    U: NDArray,
    L_hjb: Callable[[NDArray], NDArray] | sparse.spmatrix | NDArray,
    L_fp_adjoint: Callable[[NDArray], NDArray] | sparse.spmatrix | NDArray,
    dx: float | NDArray = 1.0,
    rtol: float = 1e-10,
) -> DualityVerificationResult:
    """
    Verify discrete duality: ⟨m, L_hjb(U)⟩ = ⟨L_fp*(m), U⟩.

    Uses trapezoidal quadrature for inner products.

    Args:
        m: Density field
        U: Value function field
        L_hjb: HJB operator (callable or matrix)
        L_fp_adjoint: Adjoint FP operator (callable or matrix)
        dx: Grid spacing (scalar or array for non-uniform)
        rtol: Relative tolerance

    Returns:
        DualityVerificationResult

    Example:
        >>> result = verify_duality(m, U, A_hjb, A_fp.T, dx=0.01)
        >>> print(f"Duality error: {result.relative_error:.2e}")
    """
    # Apply operators
    if callable(L_hjb):
        L_hjb_U = L_hjb(U)
    elif sparse.issparse(L_hjb):
        L_hjb_U = L_hjb @ U.flatten()
    else:
        L_hjb_U = np.asarray(L_hjb) @ U.flatten()

    if callable(L_fp_adjoint):
        L_fp_adj_m = L_fp_adjoint(m)
    elif sparse.issparse(L_fp_adjoint):
        L_fp_adj_m = L_fp_adjoint @ m.flatten()
    else:
        L_fp_adj_m = np.asarray(L_fp_adjoint) @ m.flatten()

    # Flatten for inner product
    m_flat = m.flatten()
    U_flat = U.flatten()
    L_hjb_U_flat = np.asarray(L_hjb_U).flatten()
    L_fp_adj_m_flat = np.asarray(L_fp_adj_m).flatten()

    # Trapezoidal quadrature weights
    if np.isscalar(dx):
        weights = np.ones_like(m_flat) * dx
        weights[0] = dx / 2
        weights[-1] = dx / 2
    else:
        weights = np.asarray(dx)

    # Inner products
    lhs = np.sum(m_flat * L_hjb_U_flat * weights)
    rhs = np.sum(L_fp_adj_m_flat * U_flat * weights)

    # Error
    absolute_error = abs(lhs - rhs)
    scale = max(abs(lhs), abs(rhs), 1e-15)
    relative_error = absolute_error / scale

    is_dual = relative_error < rtol

    return DualityVerificationResult(
        lhs=lhs,
        rhs=rhs,
        absolute_error=absolute_error,
        relative_error=relative_error,
        is_dual=is_dual,
    )


def compute_adjoint_error_profile(
    A_hjb: sparse.spmatrix | NDArray,
    A_fp: sparse.spmatrix | NDArray,
) -> NDArray:
    """
    Compute row-wise adjoint error profile.

    Useful for visualizing where adjoint mismatch occurs.

    Args:
        A_hjb: HJB discrete operator
        A_fp: FP discrete operator

    Returns:
        Array of shape (n,) with L2 error per row.
    """
    if sparse.issparse(A_hjb):
        A_hjb = A_hjb.toarray()
    if sparse.issparse(A_fp):
        A_fp = A_fp.toarray()

    diff = A_fp - A_hjb.T
    row_errors = np.linalg.norm(diff, axis=1)
    return row_errors
