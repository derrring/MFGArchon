"""
Monotonicity statistics tracking for numerical schemes.

This module provides utilities for tracking M-matrix property satisfaction
in finite difference and meshfree discretizations.

For a monotone scheme, the discretization matrix must be an M-matrix:
- Diagonal elements: a_ii > 0 (positive diagonal)
- Off-diagonal elements: a_ij <= 0 for i != j (non-positive off-diagonal)
- Row sums: sum_j a_ij >= 0 (weak diagonal dominance)

These conditions ensure that the maximum principle holds for the discretized PDE.

References:
    - Barles, Souganidis (1991): Convergence of approximation schemes for
      fully nonlinear second order equations.
    - Oberman (2006): Convergent difference schemes for degenerate elliptic
      and parabolic equations: Hamilton-Jacobi equations and free boundary problems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.sparse import spmatrix


class MonotonicityStats:
    """
    Track M-matrix property satisfaction statistics across a solve.

    This class records point-by-point verification of monotonicity conditions
    and computes aggregate statistics.

    Attributes:
        total_points: Number of points verified
        monotone_points: Number of points satisfying M-matrix property
        violations_by_point: Mapping from point index to list of diagnostics
        worst_violations: List of worst violations ordered by severity

    Example:
        >>> stats = MonotonicityStats()
        >>> stats.record_point(0, True, {"is_monotone": True, "violation_severity": 0.0})
        >>> stats.record_point(1, False, {"is_monotone": False, "violation_severity": 0.1})
        >>> stats.get_success_rate()
        50.0
        >>> summary = stats.get_summary()
        >>> summary["num_violating_points"]
        1
    """

    def __init__(self):
        """Initialize empty statistics tracker."""
        self.total_points = 0
        self.monotone_points = 0
        self.violations_by_point: dict[int, list[dict]] = {}
        self.worst_violations: list[dict] = []

    def record_point(self, point_idx: int, is_monotone: bool, diagnostics: dict):
        """
        Record M-matrix verification result for a single point.

        Args:
            point_idx: Index of the collocation/grid point
            is_monotone: Whether M-matrix property is satisfied
            diagnostics: Dictionary with detailed diagnostic info including:
                - is_monotone: bool
                - center_ok: bool (diagonal condition)
                - neighbors_ok: bool (off-diagonal condition)
                - w_center: float (center weight)
                - min_neighbor_weight: float
                - max_neighbor_weight: float
                - num_violations: int
                - num_neighbors: int
                - violation_severity: float (magnitude of worst violation)
        """
        self.total_points += 1
        if is_monotone:
            self.monotone_points += 1
        else:
            if point_idx not in self.violations_by_point:
                self.violations_by_point[point_idx] = []
            self.violations_by_point[point_idx].append(diagnostics)
            severity = diagnostics.get("violation_severity", 0.0)
            self.worst_violations.append({"point_idx": point_idx, "severity": severity})

    def get_success_rate(self) -> float:
        """
        Compute percentage of points satisfying M-matrix property.

        Returns:
            Success rate as percentage (0-100). Returns 0.0 if no points recorded.
        """
        if self.total_points == 0:
            return 0.0
        return 100.0 * self.monotone_points / self.total_points

    def get_summary(self) -> dict:
        """
        Get summary statistics for M-matrix verification.

        Returns:
            Dictionary with keys:
                - success_rate: Percentage of monotone points (0-100)
                - monotone_points: Count of points satisfying M-matrix property
                - total_points: Total number of points checked
                - num_violating_points: Number of distinct points with violations
                - max_violation_severity: Maximum violation magnitude
        """
        success_rate = self.get_success_rate()
        num_violating_points = len(self.violations_by_point)
        max_violation = 0.0
        if self.worst_violations:
            max_violation = max(v["severity"] for v in self.worst_violations)

        return {
            "success_rate": success_rate,
            "monotone_points": self.monotone_points,
            "total_points": self.total_points,
            "num_violating_points": num_violating_points,
            "max_violation_severity": max_violation,
        }

    def reset(self):
        """Reset all statistics to initial state."""
        self.total_points = 0
        self.monotone_points = 0
        self.violations_by_point.clear()
        self.worst_violations.clear()


def verify_m_matrix_property(
    matrix: NDArray[np.floating] | spmatrix,
    tolerance: float = 1e-12,
    check_irreducibility: bool = False,
) -> dict:
    """
    Verify M-matrix property for an assembled discretization matrix.

    An M-matrix satisfies:
    - Positive diagonal: a_ii > 0
    - Non-positive off-diagonal: a_ij <= 0 for i != j
    - Weak diagonal dominance: sum_j a_ij >= 0

    These conditions guarantee convergence to the viscosity solution
    (Barles-Souganidis 1991) and ensure the maximum principle holds.

    Args:
        matrix: Square matrix (dense ndarray or scipy sparse)
        tolerance: Numerical tolerance for comparisons (default: 1e-12)
        check_irreducibility: If True, also check matrix irreducibility
            (required for strict M-matrix, not just weak)

    Returns:
        Dictionary with diagnostic information:
        - is_m_matrix: True if all M-matrix conditions satisfied
        - positive_diagonal: True if all a_ii > 0
        - nonpositive_offdiag: True if all a_ij <= 0 for i != j
        - weak_diagonal_dominant: True if all row sums >= 0
        - num_diagonal_violations: Count of rows with a_ii <= 0
        - num_offdiag_violations: Count of positive off-diagonal entries
        - num_row_sum_violations: Count of rows with negative row sum
        - min_diagonal: Minimum diagonal value
        - max_offdiag: Maximum off-diagonal value
        - min_row_sum: Minimum row sum
        - violating_rows: List of row indices with any violation

    Example:
        >>> import numpy as np
        >>> from mfgarchon.utils.numerical.monotonicity_stats import verify_m_matrix_property
        >>> # Valid M-matrix (Laplacian-like)
        >>> A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        >>> result = verify_m_matrix_property(A)
        >>> result["is_m_matrix"]
        True
        >>> # Invalid matrix (positive off-diagonal)
        >>> B = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
        >>> result = verify_m_matrix_property(B)
        >>> result["is_m_matrix"]
        False
    """
    import scipy.sparse as sp

    # Convert to dense if sparse (for simpler indexing)
    if sp.issparse(matrix):
        A = matrix.toarray()
    else:
        A = np.asarray(matrix)

    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {A.shape}")

    # Extract diagonal
    diagonal = np.diag(A)

    # Check 1: Positive diagonal (a_ii > 0)
    diagonal_violations = diagonal <= tolerance
    num_diagonal_violations = int(np.sum(diagonal_violations))
    positive_diagonal = num_diagonal_violations == 0

    # Check 2: Non-positive off-diagonal (a_ij <= 0 for i != j)
    # Create mask for off-diagonal elements
    offdiag_mask = ~np.eye(n, dtype=bool)
    offdiag_values = A[offdiag_mask]
    offdiag_violations = offdiag_values > tolerance
    num_offdiag_violations = int(np.sum(offdiag_violations))
    nonpositive_offdiag = num_offdiag_violations == 0

    # Check 3: Weak diagonal dominance (row sums >= 0)
    row_sums = np.sum(A, axis=1)
    row_sum_violations = row_sums < -tolerance
    num_row_sum_violations = int(np.sum(row_sum_violations))
    weak_diagonal_dominant = num_row_sum_violations == 0

    # Aggregate: is M-matrix if all conditions satisfied
    is_m_matrix = positive_diagonal and nonpositive_offdiag and weak_diagonal_dominant

    # Find violating rows (any violation type)
    violating_rows = []
    for i in range(n):
        violations = []
        if diagonal[i] <= tolerance:
            violations.append("diagonal")
        if np.any(A[i, :i] > tolerance) or np.any(A[i, i + 1 :] > tolerance):
            violations.append("offdiag")
        if row_sums[i] < -tolerance:
            violations.append("row_sum")
        if violations:
            violating_rows.append({"row": i, "violations": violations})

    result = {
        "is_m_matrix": is_m_matrix,
        "positive_diagonal": positive_diagonal,
        "nonpositive_offdiag": nonpositive_offdiag,
        "weak_diagonal_dominant": weak_diagonal_dominant,
        "num_diagonal_violations": num_diagonal_violations,
        "num_offdiag_violations": num_offdiag_violations,
        "num_row_sum_violations": num_row_sum_violations,
        "min_diagonal": float(np.min(diagonal)),
        "max_offdiag": float(np.max(offdiag_values)) if len(offdiag_values) > 0 else 0.0,
        "min_row_sum": float(np.min(row_sums)),
        "violating_rows": violating_rows,
        "n": n,
    }

    # Optional: Check irreducibility (for strict M-matrix)
    if check_irreducibility:
        try:
            from scipy.sparse.csgraph import connected_components

            # Graph connectivity check: treat A as adjacency (non-zero = connected)
            adjacency = sp.csr_matrix(np.abs(A) > tolerance)
            n_components, _ = connected_components(adjacency, directed=False)
            is_irreducible = n_components == 1
            result["is_irreducible"] = is_irreducible
            result["num_components"] = int(n_components)
        except ImportError:
            result["is_irreducible"] = None
            result["num_components"] = None

    return result


def get_m_matrix_diagnostic_string(verification_result: dict) -> str:
    """
    Format M-matrix verification result as a human-readable diagnostic string.

    Args:
        verification_result: Output from verify_m_matrix_property()

    Returns:
        Formatted string with diagnostic information
    """
    lines = []
    lines.append("=" * 60)
    lines.append("M-MATRIX VERIFICATION")
    lines.append("=" * 60)

    is_m = verification_result["is_m_matrix"]
    status = "PASS" if is_m else "FAIL"
    lines.append(f"Status: {status}")
    lines.append(f"Matrix size: {verification_result['n']} x {verification_result['n']}")
    lines.append("")

    lines.append("Condition Checks:")
    lines.append(
        f"  Positive diagonal (a_ii > 0):     {'PASS' if verification_result['positive_diagonal'] else 'FAIL'}"
    )
    lines.append(
        f"  Non-positive off-diag (a_ij<=0):  {'PASS' if verification_result['nonpositive_offdiag'] else 'FAIL'}"
    )
    lines.append(
        f"  Weak diagonal dominance:          {'PASS' if verification_result['weak_diagonal_dominant'] else 'FAIL'}"
    )
    lines.append("")

    lines.append("Statistics:")
    lines.append(f"  Min diagonal value:      {verification_result['min_diagonal']:.6e}")
    lines.append(f"  Max off-diagonal value:  {verification_result['max_offdiag']:.6e}")
    lines.append(f"  Min row sum:             {verification_result['min_row_sum']:.6e}")
    lines.append("")

    if not is_m:
        lines.append("Violations:")
        lines.append(f"  Diagonal violations:     {verification_result['num_diagonal_violations']}")
        lines.append(f"  Off-diagonal violations: {verification_result['num_offdiag_violations']}")
        lines.append(f"  Row sum violations:      {verification_result['num_row_sum_violations']}")

        # Show first few violating rows
        violating = verification_result["violating_rows"]
        if violating:
            lines.append("")
            lines.append(f"First 5 violating rows (of {len(violating)} total):")
            for entry in violating[:5]:
                lines.append(f"  Row {entry['row']}: {', '.join(entry['violations'])}")

    if "is_irreducible" in verification_result:
        lines.append("")
        lines.append(f"Irreducibility: {'Yes' if verification_result['is_irreducible'] else 'No'}")
        lines.append(f"Connected components: {verification_result['num_components']}")

    lines.append("=" * 60)
    return "\n".join(lines)
