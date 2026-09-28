"""
Adjoint error diagnostics and localization.

This module provides tools for diagnosing where and why adjoint consistency
fails between HJB and FP operators.

Key functions:
- diagnose_adjoint_error: Comprehensive diagnosis
- localize_boundary_error: Identify which boundaries have issues
- suggest_fix: Recommend corrections based on error pattern

References:
-----------
- Issue #704: Unified adjoint module
- Issue #625 (state-dependent BC coupling)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class DiagnosticGeometry(Protocol):
    """Protocol for geometry objects used in diagnostics."""

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        ...

    def get_grid_shape(self) -> tuple[int, ...]:
        """Return grid shape."""
        ...


class ErrorSource(Enum):
    """Classification of adjoint error source."""

    NONE = "none"
    INTERIOR = "interior"
    BOUNDARY = "boundary"
    MIXED = "mixed"


class ErrorSeverity(Enum):
    """Severity of adjoint error."""

    OK = "ok"  # < 1e-10
    LOW = "low"  # 1e-10 to 1e-6
    MEDIUM = "medium"  # 1e-6 to 1e-2
    HIGH = "high"  # > 1e-2


@dataclass
class BoundaryErrorInfo:
    """Error information for a specific boundary."""

    name: str
    """Boundary name (e.g., 'x_min', 'x_max')."""

    indices: list[int]
    """Indices of this boundary in the matrix."""

    error: float
    """L2 error at this boundary."""

    max_error: float
    """Maximum absolute error at this boundary."""

    max_error_idx: int
    """Index of maximum error within this boundary."""


@dataclass
class AdjointDiagnosticReport:
    """Comprehensive diagnostic report for adjoint errors."""

    # Overall assessment
    error_source: ErrorSource
    """Primary source of error."""

    severity: ErrorSeverity
    """Overall severity."""

    total_error: float
    """Total relative error."""

    # Decomposition
    interior_error: float
    """Error from interior points."""

    boundary_error: float
    """Error from boundary points."""

    interior_fraction: float
    """Fraction of error from interior."""

    boundary_fraction: float
    """Fraction of error from boundary."""

    # Boundary-specific
    boundary_details: list[BoundaryErrorInfo] = field(default_factory=list)
    """Detailed error for each boundary."""

    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    """Suggested fixes."""

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "ADJOINT DIAGNOSTIC REPORT",
            "=" * 60,
            f"Error Source: {self.error_source.value}",
            f"Severity: {self.severity.value}",
            f"Total Error: {self.total_error:.2e}",
            "",
            "Error Decomposition:",
            f"  Interior: {self.interior_error:.2e} ({self.interior_fraction:.1%})",
            f"  Boundary: {self.boundary_error:.2e} ({self.boundary_fraction:.1%})",
        ]

        if self.boundary_details:
            lines.append("")
            lines.append("Boundary Details:")
            for bd in self.boundary_details:
                lines.append(f"  {bd.name}: error={bd.error:.2e}, max={bd.max_error:.2e}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)


def diagnose_adjoint_error(
    A_hjb: sparse.spmatrix | NDArray,
    A_fp: sparse.spmatrix | NDArray,
    geometry: DiagnosticGeometry | None = None,
    *,
    dimension: int = 1,
    grid_shape: tuple[int, ...] | None = None,
    boundary_names: list[str] | None = None,
) -> AdjointDiagnosticReport:
    """
    Comprehensive diagnosis of adjoint error.

    Args:
        A_hjb: HJB discrete operator
        A_fp: FP discrete operator
        geometry: Optional geometry object (preferred). If provided, extracts
            dimension and grid_shape automatically.
        dimension: Spatial dimension (used if geometry not provided)
        grid_shape: Shape of the spatial grid (used if geometry not provided)
        boundary_names: Names for boundaries (default: x_min, x_max, ...)

    Returns:
        AdjointDiagnosticReport with detailed analysis.

    Example:
        >>> # Preferred: use geometry
        >>> from mfgarchon.geometry import TensorProductGrid
        >>> grid = TensorProductGrid(bounds=[(0, 1)], Nx=[50])
        >>> report = diagnose_adjoint_error(A_hjb, A_fp, geometry=grid)

        >>> # Alternative: specify dimension directly
        >>> report = diagnose_adjoint_error(A_hjb, A_fp, dimension=1)
        >>> print(report)
    """
    # Convert to dense
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
    total_error = frobenius_error / hjb_norm if hjb_norm > 0 else frobenius_error

    # Extract dimension and grid_shape from geometry if provided
    if geometry is not None:
        dimension = geometry.dimension
        grid_shape = geometry.get_grid_shape()
    elif grid_shape is None:
        if dimension == 1:
            grid_shape = (n,)
        else:
            # Assume square grid for higher dimensions
            side = round(n ** (1.0 / dimension))
            grid_shape = tuple([side] * dimension)

    # Identify boundary indices
    boundary_indices_dict = _get_boundary_indices(grid_shape, dimension, boundary_names)
    all_boundary = set()
    for indices in boundary_indices_dict.values():
        all_boundary.update(indices)
    interior_indices = [i for i in range(n) if i not in all_boundary]

    # Interior error
    if len(interior_indices) > 0:
        interior_mask = np.ix_(interior_indices, interior_indices)
        interior_diff = diff[interior_mask]
        interior_error = np.linalg.norm(interior_diff, "fro")
    else:
        interior_error = 0.0

    # Boundary error
    boundary_error = np.sqrt(max(0, frobenius_error**2 - interior_error**2))

    # Fractions
    total_sq = frobenius_error**2 if frobenius_error > 0 else 1.0
    interior_fraction = (interior_error**2) / total_sq
    boundary_fraction = (boundary_error**2) / total_sq

    # Boundary details
    boundary_details = []
    for name, indices in boundary_indices_dict.items():
        if len(indices) == 0:
            continue
        # Error at this boundary (rows)
        row_errors = np.linalg.norm(diff[indices, :], axis=1)
        bd_error = np.linalg.norm(row_errors)
        max_idx = np.argmax(row_errors)
        boundary_details.append(
            BoundaryErrorInfo(
                name=name,
                indices=list(indices),
                error=bd_error,
                max_error=row_errors[max_idx],
                max_error_idx=indices[max_idx],
            )
        )

    # Classify error source
    if total_error < 1e-10:
        error_source = ErrorSource.NONE
    elif boundary_fraction > 0.9:
        error_source = ErrorSource.BOUNDARY
    elif interior_fraction > 0.9:
        error_source = ErrorSource.INTERIOR
    else:
        error_source = ErrorSource.MIXED

    # Classify severity
    if total_error < 1e-10:
        severity = ErrorSeverity.OK
    elif total_error < 1e-6:
        severity = ErrorSeverity.LOW
    elif total_error < 1e-2:
        severity = ErrorSeverity.MEDIUM
    else:
        severity = ErrorSeverity.HIGH

    # Generate recommendations
    recommendations = _generate_recommendations(error_source, severity, boundary_details, total_error)

    return AdjointDiagnosticReport(
        error_source=error_source,
        severity=severity,
        total_error=total_error,
        interior_error=interior_error,
        boundary_error=boundary_error,
        interior_fraction=interior_fraction,
        boundary_fraction=boundary_fraction,
        boundary_details=boundary_details,
        recommendations=recommendations,
    )


def _get_boundary_indices(
    grid_shape: tuple[int, ...],
    dimension: int,
    boundary_names: list[str] | None = None,
) -> dict[str, list[int]]:
    """
    Get boundary indices for each boundary face (truly dimension-agnostic).

    Uses geometry module convention: x_min, x_max, y_min, y_max, z_min, z_max, ...
    Each boundary face contains all points on that face (including corners/edges).
    Points on edges/corners belong to multiple boundaries.

    Args:
        grid_shape: Grid shape tuple (N0, N1, ..., Nd-1)
        dimension: Spatial dimension
        boundary_names: Override default boundary names

    Returns:
        Dict mapping boundary names to flat indices (row-major/C ordering)
    """
    # Axis naming convention:
    # - dims ≤ 3: x, y, z
    # - dims > 3: x_1, x_2, ..., x_d (for ALL dimensions)
    if boundary_names is None:
        boundary_names = []
        for d in range(dimension):
            axis = _get_axis_name(d, dimension)
            boundary_names.extend([f"{axis}_min", f"{axis}_max"])

    result = {name: [] for name in boundary_names}

    # Total number of points
    N_total = int(np.prod(grid_shape))

    # Iterate over all points and check which boundaries they belong to
    for flat_idx in range(N_total):
        # Convert flat index to multi-index (row-major/C order)
        multi_idx = _flat_to_multi_index(flat_idx, grid_shape)

        # Check each dimension
        for d in range(dimension):
            axis = _get_axis_name(d, dimension)
            if multi_idx[d] == 0:
                result[f"{axis}_min"].append(flat_idx)
            if multi_idx[d] == grid_shape[d] - 1:
                result[f"{axis}_max"].append(flat_idx)

    return result


def _flat_to_multi_index(flat_idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Convert flat index to multi-index (row-major/C order)."""
    multi_idx = []
    remaining = flat_idx
    for i, _dim_size in enumerate(shape):
        stride = int(np.prod(shape[i + 1 :])) if i + 1 < len(shape) else 1
        idx = remaining // stride
        remaining = remaining % stride
        multi_idx.append(idx)
    return tuple(multi_idx)


def _get_axis_name(dim_index: int, total_dims: int) -> str:
    """
    Get axis name for dimension index.

    Convention:
    - If total_dims ≤ 3: use x, y, z
    - If total_dims > 3: use x_1, x_2, ..., x_d for ALL dimensions
    """
    if total_dims <= 3:
        return ["x", "y", "z"][dim_index]
    else:
        return f"x_{dim_index + 1}"


def _generate_recommendations(
    error_source: ErrorSource,
    severity: ErrorSeverity,
    boundary_details: list[BoundaryErrorInfo],
    total_error: float,
) -> list[str]:
    """Generate fix recommendations based on error pattern."""
    recs = []

    if severity == ErrorSeverity.OK:
        recs.append("Adjoint consistency is satisfied. No action needed.")
        return recs

    if error_source == ErrorSource.BOUNDARY:
        recs.append(
            "Error is concentrated at boundaries. Check that the HJB and FP boundary rows are the "
            "paired discretisation of one wall condition."
        )

        # Find worst boundary
        if boundary_details:
            worst = max(boundary_details, key=lambda x: x.error)
            recs.append(f"Worst boundary: {worst.name} (error={worst.error:.2e})")

    elif error_source == ErrorSource.INTERIOR:
        recs.append("Error is in interior points. Check that HJB and FP use the same upwind stencils.")
        recs.append("Verify advection discretization matches between solvers.")

    else:  # MIXED
        recs.append("Error in both interior and boundary. Multiple issues may exist.")
        recs.append("1. First fix boundary BC consistency")
        recs.append("2. Then verify interior stencil matching")

    if severity == ErrorSeverity.HIGH:
        recs.append(f"WARNING: Error is high ({total_error:.2e}). MFG convergence may be compromised.")

    return recs


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    """Quick validation of diagnostic utilities."""
    print("Testing adjoint diagnostic utilities...")
    print()

    n = 20

    # Test 1: Perfect adjoint
    print("Test 1: Perfect adjoint (symmetric matrix)")
    A_sym = np.random.randn(n, n)
    A_sym = (A_sym + A_sym.T) / 2

    report = diagnose_adjoint_error(A_sym, A_sym, dimension=1)
    print(report)
    assert report.severity == ErrorSeverity.OK
    print()

    # Test 2: Boundary error only
    print("Test 2: Boundary error simulation")
    A_hjb = np.eye(n) * 2 - np.eye(n, k=1) - np.eye(n, k=-1)
    A_fp = A_hjb.T.copy()
    # Add boundary perturbation
    A_fp[0, 0] += 0.1
    A_fp[-1, -1] += 0.1

    report = diagnose_adjoint_error(A_hjb, A_fp, dimension=1)
    print(report)
    assert report.error_source == ErrorSource.BOUNDARY
    print()

    # Test 3: Interior error
    print("Test 3: Interior error simulation")
    A_hjb = np.eye(n) * 2 - np.eye(n, k=1) - np.eye(n, k=-1)
    A_fp = A_hjb.T.copy()
    # Add interior perturbation
    A_fp[n // 2, n // 2] += 0.5
    A_fp[n // 2 + 1, n // 2 + 1] += 0.5

    report = diagnose_adjoint_error(A_hjb, A_fp, dimension=1)
    print(report)
    print()

    print("All diagnostic tests passed!")
