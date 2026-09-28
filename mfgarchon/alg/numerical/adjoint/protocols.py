"""
Protocols for adjoint-capable solvers.

Two approaches for adjoint consistency in MFG coupling:

1. **Scheme pairing** (Issue #706): Use matched HJB/FP schemes
   (gradient_upwind + divergence_upwind). Default and simplest approach.

2. **True adjoint via Jacobian transpose** (Issue #707): Build the linearized
   HJB operator (Jacobian) and transpose it to get the FP operator.
   Correct for arbitrary Hamiltonians, not just quadratic H = (c/2)|p|^2.
   Requires class-based Hamiltonian with dp() method.

Note: The old "transpose" mode (Issue #622, deprecated in #706) naively
transposed the velocity-based advection matrix. This is incorrect for
non-symmetric upwind stencils. The "jacobian_transpose" mode (#707) is
the correct implementation.

See Issue #706 (adjoint discretization) for mathematical foundations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mfgarchon.utils.deprecation import deprecated

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from scipy import sparse


@runtime_checkable
class AdjointCapableHJBSolver(Protocol):
    """
    DEPRECATED: Protocol for HJB solvers that support strict adjoint mode.

    This protocol is deprecated because the underlying approach (transposing
    the gradient matrix) is mathematically incorrect. Use scheme pairing instead:
    - HJB: gradient_upwind + FP: divergence_upwind

    See Issue #706 (adjoint discretization).
    """

    def build_advection_matrix(
        self,
        U: NDArray[np.floating],
        time_index: int | None = None,
    ) -> sparse.csr_matrix:
        """
        DEPRECATED: Build advection matrix for given value function U.

        This method is deprecated because simple matrix transpose does NOT
        give the correct FP operator. Use divergence_upwind FP scheme instead.
        """
        ...


@runtime_checkable
class AdjointCapableFPSolver(Protocol):
    """
    DEPRECATED: Protocol for FP solvers that support strict adjoint mode.

    This protocol is deprecated because the underlying approach (using A_hjb.T)
    is mathematically incorrect. Use divergence_upwind scheme instead.

    See Issue #706 (adjoint discretization).
    """

    def solve_fp_step_adjoint_mode(
        self,
        m_current: NDArray[np.floating],
        A_advection_T: sparse.csr_matrix,
        volatility: float,
        volatility_kind: str | None = None,
    ) -> NDArray[np.floating]:
        """
        DEPRECATED: Solve one FP timestep using externally-provided advection matrix.

        This method is deprecated because using A_hjb.T is mathematically incorrect.
        Use standard solve_fp_step with divergence_upwind scheme instead.
        """
        ...


@runtime_checkable
class LinearizedOperatorCapable(Protocol):
    """
    Protocol for HJB solvers that support true adjoint mode (Issue #707).

    Solvers implementing this protocol can build the linearized HJB operator
    (Jacobian) whose transpose gives the correct FP advection operator for
    arbitrary Hamiltonians.
    """

    def build_linearized_operator(
        self,
        U: NDArray[np.floating],
        M: NDArray[np.floating],
        time: float = 0.0,
    ) -> sparse.csr_matrix:
        """
        Build the linearized HJB advection operator.

        Returns A_adv[i,j] = (dH/dp)_i * (dp_i/dU_j).
        Transpose A_adv^T is the correct FP advection operator.
        """
        ...


@deprecated(
    since="v0.17.0",
    replacement="validate_scheme_pairing",
    reason="The adjoint_mode approach (using A_hjb.T for FP) is mathematically incorrect. "
    "Use scheme pairing instead: gradient_upwind + divergence_upwind. See Issue #706.",
)
def validate_adjoint_capability(
    hjb_solver: object,
    fp_solver: object,
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """
    DEPRECATED: Validate that solvers support strict adjoint mode.

    This function is deprecated because the adjoint mode approach is mathematically
    incorrect. Use scheme pairing validation instead:

    ```python
    def validate_scheme_pairing(hjb_solver, fp_solver) -> bool:
        valid_pairs = {
            "gradient_upwind": "divergence_upwind",
            "gradient_centered": "divergence_centered",
        }
        hjb_scheme = getattr(hjb_solver, 'advection_scheme', None)
        fp_scheme = getattr(fp_solver, 'advection_scheme', None)
        return valid_pairs.get(hjb_scheme) == fp_scheme
    ```

    See Issue #706 (adjoint discretization).
    """
    issues = []

    if not isinstance(hjb_solver, AdjointCapableHJBSolver):
        issues.append(
            f"HJB solver {type(hjb_solver).__name__} does not implement "
            "AdjointCapableHJBSolver protocol (missing build_advection_matrix)"
        )

    if not isinstance(fp_solver, AdjointCapableFPSolver):
        issues.append(
            f"FP solver {type(fp_solver).__name__} does not implement "
            "AdjointCapableFPSolver protocol (missing solve_fp_step_adjoint_mode)"
        )

    is_valid = len(issues) == 0

    if strict and not is_valid:
        raise TypeError(
            "Solvers do not support strict adjoint mode:\n  - "
            + "\n  - ".join(issues)
            + "\n\nNOTE: This validation is deprecated. The adjoint_mode approach "
            "is mathematically incorrect. Use scheme pairing instead."
        )

    return is_valid, issues


def validate_scheme_pairing(hjb_solver: object, fp_solver: object) -> tuple[bool, str | None]:
    """
    Validate that HJB and FP solvers use mathematically compatible schemes.

    This is the CORRECT way to ensure discrete adjoint consistency in MFG systems.
    The FP divergence scheme must match the HJB gradient scheme.

    Args:
        hjb_solver: HJB solver instance
        fp_solver: FP solver instance

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if schemes are correctly paired
        - error_message: None if valid, otherwise describes the issue

    Example:
        >>> is_valid, error = validate_scheme_pairing(hjb_solver, fp_solver)
        >>> if not is_valid:
        ...     raise ValueError(error)

    Valid Pairings:
        - gradient_upwind + divergence_upwind (stability-first)
        - gradient_centered + divergence_centered (accuracy-first)

    See Issue #706 (adjoint discretization) for mathematical foundations.
    """
    valid_pairs = {
        "gradient_upwind": "divergence_upwind",
        "gradient_centered": "divergence_centered",
    }

    hjb_scheme = getattr(hjb_solver, "advection_scheme", None)
    fp_scheme = getattr(fp_solver, "advection_scheme", None)

    if hjb_scheme is None:
        return False, f"HJB solver {type(hjb_solver).__name__} has no advection_scheme attribute"

    if fp_scheme is None:
        return False, f"FP solver {type(fp_solver).__name__} has no advection_scheme attribute"

    expected_fp_scheme = valid_pairs.get(hjb_scheme)
    if expected_fp_scheme is None:
        return False, f"Unknown HJB scheme: {hjb_scheme}. Expected one of {list(valid_pairs.keys())}"

    if fp_scheme != expected_fp_scheme:
        return False, (
            f"Scheme mismatch: HJB uses '{hjb_scheme}' but FP uses '{fp_scheme}'. "
            f"Expected FP to use '{expected_fp_scheme}' for correct adjoint structure."
        )

    return True, None


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    """Quick validation of protocol definitions."""
    print("Testing adjoint protocols (DEPRECATED)...")
    print()

    # Test new validate_scheme_pairing
    print("Test: validate_scheme_pairing")

    class MockHJBSolver:
        advection_scheme = "gradient_upwind"

    class MockFPSolverCorrect:
        advection_scheme = "divergence_upwind"

    class MockFPSolverWrong:
        advection_scheme = "gradient_upwind"  # Wrong!

    # Valid pairing
    is_valid, error = validate_scheme_pairing(MockHJBSolver(), MockFPSolverCorrect())
    assert is_valid, f"Should be valid: {error}"
    print("  Valid pairing (gradient_upwind + divergence_upwind): PASSED")

    # Invalid pairing
    is_valid, error = validate_scheme_pairing(MockHJBSolver(), MockFPSolverWrong())
    assert not is_valid, "Should be invalid"
    assert "divergence_upwind" in error
    print("  Invalid pairing detected: PASSED")
    print()

    print("All tests passed!")
