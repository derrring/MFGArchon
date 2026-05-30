"""
Adjoint Operator Duality Validation for MFG Solver Pairing.

This module provides validation logic for ensuring HJB-FP solver pairs satisfy
the required duality relationships for convergent MFG schemes (Issue #580).

Mathematical Background
-----------------------

Mean Field Games require that the HJB operator L_HJB and FP operator L_FP
satisfy an adjoint relationship. There are two types of duality:

**Type A: Discrete Duality (Exact)**
    L_FP = L_HJB^T at the matrix level

    Families: FDM, Semi-Lagrangian, FVM

    Example: For FDM upwind, if HJB uses backward differences for -grad(u),
    then FP must use forward differences for div(m*alpha) such that the
    discrete operators are exact transposes.

**Type B: Continuous Duality (Asymptotic)**
    L_FP = L_HJB^T + O(h)

    Families: GFDM, PINN

    Example: GFDM uses asymmetric neighborhoods, so discrete transpose is not
    exact. However, the continuous operators are adjoints, giving O(h) error.
    Requires renormalization for Nash gap convergence.

Validation Logic
----------------

The check_solver_duality() function uses _scheme_family traits to classify
solver pairs:

1. **Same family, Type A** → DISCRETE_DUAL (exact transpose)
2. **Same family, Type B** → CONTINUOUS_DUAL (asymptotic transpose)
3. **Different families** → NOT_DUAL (mixing FDM with GFDM breaks duality)
4. **Missing trait** → VALIDATION_SKIPPED (unannotated solvers)

References
----------
- Issue #580: Adjoint-aware solver pairing
- Issue #543: Validator pattern (getattr instead of hasattr)
- Issue #706: Adjoint operators mathematical theory
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any

from mfgarchon.alg.base_solver import SchemeFamily


class DualityStatus(Enum):
    """
    Classification of HJB-FP duality relationship.

    Values:
        DISCRETE_DUAL: Exact discrete transpose (L_FP = L_HJB^T)
        CONTINUOUS_DUAL: Asymptotic transpose (L_FP = L_HJB^T + O(h))
        NOT_DUAL: Different scheme families (duality broken)
        VALIDATION_SKIPPED: Cannot validate (missing traits)
    """

    DISCRETE_DUAL = "discrete_dual"  # Type A: FDM, SL, FVM
    CONTINUOUS_DUAL = "continuous_dual"  # Type B: GFDM, PINN
    NOT_DUAL = "not_dual"  # Mixed families
    VALIDATION_SKIPPED = "validation_skipped"  # Missing _scheme_family


class DualityValidationResult:
    """
    Result of duality validation with metadata and recommendations.

    Attributes:
        status: DualityStatus classification
        hjb_family: Scheme family of HJB solver (or None)
        fp_family: Scheme family of FP solver (or None)
        message: Human-readable explanation
        recommendation: Suggested action (or None)
    """

    def __init__(
        self,
        status: DualityStatus,
        hjb_family: SchemeFamily | None,
        fp_family: SchemeFamily | None,
        message: str,
        recommendation: str | None = None,
    ):
        self.status = status
        self.hjb_family = hjb_family
        self.fp_family = fp_family
        self.message = message
        self.recommendation = recommendation

    def __repr__(self) -> str:
        return (
            f"DualityValidationResult(status={self.status.value}, "
            f"hjb_family={self.hjb_family}, fp_family={self.fp_family})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = [f"Status: {self.status.value}"]
        if self.hjb_family is not None:
            parts.append(f"HJB: {self.hjb_family.value}")
        if self.fp_family is not None:
            parts.append(f"FP: {self.fp_family.value}")
        parts.append(self.message)
        if self.recommendation:
            parts.append(f"Recommendation: {self.recommendation}")
        return " | ".join(parts)

    def is_valid_pairing(self) -> bool:
        """Check if this is a valid dual pairing (Type A or Type B)."""
        return self.status in (DualityStatus.DISCRETE_DUAL, DualityStatus.CONTINUOUS_DUAL)

    def requires_renormalization(self) -> bool:
        """Check if this pairing requires renormalization for convergence."""
        return self.status == DualityStatus.CONTINUOUS_DUAL


def check_solver_duality(
    hjb_solver: type | Any,
    fp_solver: type | Any,
    warn_on_mismatch: bool = True,
) -> DualityValidationResult:
    """
    Validate duality relationship between HJB and FP solvers.

    This function checks if two solvers form a valid adjoint pair for MFG
    convergence. It uses _scheme_family traits (Issue #543 validator pattern)
    to determine the duality type without fragile class name matching.

    Args:
        hjb_solver: HJB solver class or instance
        fp_solver: FP solver class or instance
        warn_on_mismatch: If True, emit warnings for NOT_DUAL pairings

    Returns:
        DualityValidationResult with status, families, and recommendations

    Examples:
        >>> from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
        >>> from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        >>> result = check_solver_duality(HJBFDMSolver, FPFDMSolver)
        >>> assert result.is_valid_pairing()
        >>> assert result.status == DualityStatus.DISCRETE_DUAL

        >>> from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
        >>> from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver
        >>> result = check_solver_duality(HJBGFDMSolver, FPGFDMSolver)
        >>> assert result.is_valid_pairing()
        >>> assert result.requires_renormalization()  # Type B needs renorm
    """
    # Extract solver classes if instances were passed
    hjb_class = hjb_solver if isinstance(hjb_solver, type) else type(hjb_solver)
    fp_class = fp_solver if isinstance(fp_solver, type) else type(fp_solver)

    # Extract scheme families using validator pattern (Issue #543)
    hjb_family = getattr(hjb_class, "_scheme_family", None)
    fp_family = getattr(fp_class, "_scheme_family", None)

    # Case 1: Either solver missing _scheme_family trait → skip validation
    if hjb_family is None or fp_family is None:
        missing = []
        if hjb_family is None:
            missing.append(hjb_class.__name__)
        if fp_family is None:
            missing.append(fp_class.__name__)

        return DualityValidationResult(
            status=DualityStatus.VALIDATION_SKIPPED,
            hjb_family=hjb_family,
            fp_family=fp_family,
            message=f"Cannot validate duality: {', '.join(missing)} missing _scheme_family trait",
            recommendation="Add _scheme_family = SchemeFamily.XXX to solver class (Issue #580)",
        )

    # Case 2: Either solver is GENERIC → skip validation
    if hjb_family == SchemeFamily.GENERIC or fp_family == SchemeFamily.GENERIC:
        return DualityValidationResult(
            status=DualityStatus.VALIDATION_SKIPPED,
            hjb_family=hjb_family,
            fp_family=fp_family,
            message="Validation skipped: GENERIC scheme family does not guarantee duality",
            recommendation="Use annotated scheme families (FDM, SL, GFDM) for validated pairings",
        )

    # Case 3: Different families → NOT_DUAL
    if hjb_family != fp_family:
        msg = (
            f"Solver pair not dual: {hjb_class.__name__} ({hjb_family.value}) "
            f"paired with {fp_class.__name__} ({fp_family.value})"
        )

        if warn_on_mismatch:
            warnings.warn(
                f"\n{'=' * 70}\n"
                f"DUALITY MISMATCH WARNING\n"
                f"{'=' * 70}\n"
                f"{msg}\n\n"
                f"Mixing scheme families breaks adjoint duality, leading to:\n"
                f"  • Non-zero Nash gap even as h→0\n"
                f"  • Poor convergence or divergence\n"
                f"  • Violation of MFG equilibrium conditions\n\n"
                f"Recommendation: Use matching families:\n"
                f"  • HJBFDMSolver ↔ FPFDMSolver (discrete duality)\n"
                f"  • HJBSemiLagrangianSolver ↔ FPSLSolver (discrete duality)\n"
                f"  • HJBGFDMSolver ↔ FPGFDMSolver (continuous duality, needs renorm)\n"
                f"{'=' * 70}\n",
                UserWarning,
                stacklevel=2,
            )

        return DualityValidationResult(
            status=DualityStatus.NOT_DUAL,
            hjb_family=hjb_family,
            fp_family=fp_family,
            message=msg,
            recommendation="Use solvers from the same scheme family for proper duality",
        )

    # Case 4: Same family → Check duality type
    # Type A families: Discrete transpose (exact). MESHLESS_GALERKIN is the
    # meshfree Type-A member (Galerkin MLS, A_FP = A_HJB^T exact; Issue #1131).
    discrete_families = {
        SchemeFamily.FDM,
        SchemeFamily.SL,
        SchemeFamily.FVM,
        SchemeFamily.MESHLESS_GALERKIN,
    }

    if hjb_family in discrete_families:
        # FDM: the default upwind pairing (gradient_upwind HJB + divergence_upwind FP) is
        # structure-preserving (both mass-conservative, consistent) but the FP operator is
        # assembled independently and is NOT a bit-exact transpose in general; the exact
        # L_FP = L_HJB^T is opt-in via the iterator's adjoint_mode="jacobian_transpose".
        # SL (splatting = transpose of interpolation, #708) and MESHLESS_GALERKIN (Galerkin
        # MLS, #1131) are exact transposes by construction.
        # NOTE: this check matches only the _scheme_family enum, not the actual advection
        # scheme pairing, so it does not detect a mismatched same-family pair.
        if hjb_family == SchemeFamily.FDM:
            message = (
                "Valid discrete dual pair: FDM schemes (structure-preserving; exact "
                "L_FP = L_HJB^T is opt-in via adjoint_mode='jacobian_transpose')"
            )
        else:
            message = f"Valid discrete dual pair: {hjb_family.value} schemes (L_FP = L_HJB^T)"
        return DualityValidationResult(
            status=DualityStatus.DISCRETE_DUAL,
            hjb_family=hjb_family,
            fp_family=fp_family,
            message=message,
            recommendation=None,  # No action needed
        )

    # Type B families: Continuous transpose (asymptotic)
    continuous_families = {SchemeFamily.GFDM, SchemeFamily.PINN}

    if hjb_family in continuous_families:
        return DualityValidationResult(
            status=DualityStatus.CONTINUOUS_DUAL,
            hjb_family=hjb_family,
            fp_family=fp_family,
            message=f"Valid continuous dual pair: {hjb_family.value} schemes (L_FP = L_HJB^T + O(h))",
            recommendation="Enable renormalization for Nash gap convergence (automatic in Phase 2 factory)",
        )

    # Fallback: Unknown family (should not reach here if enum is complete)
    return DualityValidationResult(
        status=DualityStatus.VALIDATION_SKIPPED,
        hjb_family=hjb_family,
        fp_family=fp_family,
        message=f"Unknown scheme family: {hjb_family.value}",
        recommendation="Report this as a bug (Issue #580)",
    )


def validate_scheme_config(
    scheme: str | Any,
    hjb_solver: type | Any,
    fp_solver: type | Any,
) -> DualityValidationResult:
    """
    Validate that a NumericalScheme matches the actual solver pair.

    This function is used in Phase 3 when users specify a scheme via Safe Mode
    (problem.solve(scheme=NumericalScheme.FDM_UPWIND)) but manually inject
    mismatched solvers via Expert Mode.

    Args:
        scheme: NumericalScheme enum value or string
        hjb_solver: HJB solver class or instance
        fp_solver: FP solver class or instance

    Returns:
        DualityValidationResult indicating if solvers match the scheme

    Examples:
        >>> from mfgarchon.types import NumericalScheme
        >>> from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
        >>> from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver
        >>> result = validate_scheme_config(
        ...     NumericalScheme.FDM_UPWIND, HJBFDMSolver, FPGFDMSolver
        ... )
        >>> assert result.status == DualityStatus.NOT_DUAL  # Mismatch!
    """
    from mfgarchon.types import NumericalScheme

    # Convert string to enum if needed
    if isinstance(scheme, str):
        try:
            scheme = NumericalScheme[scheme.upper()]
        except KeyError:
            return DualityValidationResult(
                status=DualityStatus.VALIDATION_SKIPPED,
                hjb_family=None,
                fp_family=None,
                message=f"Unknown scheme: {scheme}",
                recommendation="Use valid NumericalScheme enum values",
            )

    # Check solver pair duality
    result = check_solver_duality(hjb_solver, fp_solver, warn_on_mismatch=False)

    # If not dual, add scheme context to message
    if result.status == DualityStatus.NOT_DUAL:
        result.message = (
            f"Scheme {scheme.value} expects matching families, but solvers are "
            f"{result.hjb_family.value} (HJB) and {result.fp_family.value} (FP)"
        )
        result.recommendation = "Use scheme-appropriate solvers or omit scheme parameter"

    return result
