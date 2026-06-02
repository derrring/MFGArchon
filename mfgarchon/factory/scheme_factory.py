"""
Scheme-Based Solver Factory for MFG Problems.

This module provides automatic creation of validated HJB-FP solver pairs based
on NumericalScheme selection (Issue #580). It ensures adjoint duality by using
_scheme_family trait matching and provides educational feedback for users.

Usage (Safe Mode - Phase 3):
    >>> from mfgarchon import MFGProblem, NumericalScheme
    >>> from mfgarchon.factory import create_paired_solvers
    >>>
    >>> problem = MFGProblem(...)
    >>> hjb, fp = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)
    >>> # hjb and fp are guaranteed to be dual by construction

Benefits over Manual Solver Selection:
    - Automatic duality guarantee (no mixing FDM with GFDM)
    - Config threading to both solvers
    - Scheme-appropriate defaults (e.g., renormalization for GFDM)
    - Educational warnings for advanced users overriding defaults

References:
    - Issue #580: Adjoint-aware solver pairing
    - Issue #543: Validator pattern (getattr instead of hasattr)
    - Issue #706: Adjoint operators mathematical theory
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mfgarchon.types import NumericalScheme
from mfgarchon.utils import DualityStatus, check_solver_duality

if TYPE_CHECKING:
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.mfg_problem import MFGProblem


def create_paired_solvers(
    problem: MFGProblem,
    scheme: NumericalScheme,
    hjb_config: dict[str, Any] | None = None,
    fp_config: dict[str, Any] | None = None,
    validate_duality: bool = True,
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """
    Create a validated HJB-FP solver pair for a given numerical scheme.

    This factory function automatically selects the correct solver classes based
    on the specified NumericalScheme, ensuring they form a valid adjoint pair.
    Configuration parameters are threaded to both solvers.

    Args:
        problem: MFG problem instance to solve
        scheme: Numerical discretization scheme (FDM_UPWIND, SL_LINEAR, GFDM, etc.)
        hjb_config: Optional configuration dict for HJB solver initialization
        fp_config: Optional configuration dict for FP solver initialization
        validate_duality: If True, validate solver duality (recommended)

    Returns:
        Tuple of (hjb_solver, fp_solver) instances that form a dual pair

    Raises:
        ValueError: If scheme is not recognized or solvers are not dual
        NotImplementedError: If scheme is defined but not yet implemented

    Examples:
        >>> # Safe Mode: Automatic dual pairing
        >>> hjb, fp = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)
        >>> assert hjb._scheme_family == fp._scheme_family  # Guaranteed

        >>> # With custom configs
        >>> hjb, fp = create_paired_solvers(
        ...     problem,
        ...     NumericalScheme.GFDM,
        ...     hjb_config={"delta": 0.05},
        ...     fp_config={"delta": 0.05, "upwind_scheme": "exponential"},
        ... )

        >>> # GFDM automatically gets renormalization recommendation
        >>> result = check_solver_duality(hjb, fp)
        >>> assert result.requires_renormalization()  # Type B scheme
    """
    # Initialize configs
    hjb_config = hjb_config or {}
    fp_config = fp_config or {}

    # Route to scheme-specific factory
    if scheme in (NumericalScheme.FDM_UPWIND, NumericalScheme.FDM_CENTERED):
        hjb_solver, fp_solver = _create_fdm_pair(problem, scheme, hjb_config, fp_config)

    elif scheme in (NumericalScheme.SL_LINEAR, NumericalScheme.SL_CUBIC):
        hjb_solver, fp_solver = _create_sl_pair(problem, scheme, hjb_config, fp_config)

    elif scheme == NumericalScheme.GFDM:
        hjb_solver, fp_solver = _create_gfdm_pair(problem, hjb_config, fp_config)

    elif scheme in (NumericalScheme.FEM_P1, NumericalScheme.FEM_P2):
        hjb_solver, fp_solver = _create_fem_pair(problem, scheme, hjb_config, fp_config)

    elif scheme == NumericalScheme.MESHLESS_GALERKIN:
        hjb_solver, fp_solver = _create_meshless_galerkin_pair(problem, hjb_config, fp_config)

    else:
        raise NotImplementedError(
            f"Scheme {scheme.value} is defined but not yet implemented in factory. "
            f"Available schemes: FDM_UPWIND, FDM_CENTERED, SL_LINEAR, GFDM, FEM_P1, FEM_P2"
        )

    # Validate duality
    if validate_duality:
        result = check_solver_duality(hjb_solver, fp_solver, warn_on_mismatch=True)

        if result.status == DualityStatus.NOT_DUAL:
            raise ValueError(
                f"Factory created non-dual solver pair (this is a bug!):\n"
                f"  HJB: {type(hjb_solver).__name__} ({result.hjb_family})\n"
                f"  FP: {type(fp_solver).__name__} ({result.fp_family})\n"
                f"Please report this as Issue #580 bug."
            )

        if result.status == DualityStatus.VALIDATION_SKIPPED:
            raise ValueError(
                f"Factory created solvers without _scheme_family traits (this is a bug!):\n"
                f"  HJB: {type(hjb_solver).__name__}\n"
                f"  FP: {type(fp_solver).__name__}\n"
                f"Please report this as Issue #580 bug."
            )

    return hjb_solver, fp_solver


def _create_fdm_pair(
    problem: MFGProblem,
    scheme: NumericalScheme,
    hjb_config: dict[str, Any],
    fp_config: dict[str, Any],
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """
    Create FDM HJB-FP solver pair.

    Discrete duality (Type A): the default pairing (gradient_upwind HJB +
    divergence_upwind FP) is structure-preserving -- both are mass-conservative
    and consistent with the continuous adjoint -- but the FP operator is assembled
    independently and is NOT a bit-exact transpose of the HJB operator in general
    (the two candidate HJB operators, the velocity advection and the linearized
    Jacobian, are themselves different). The exact discrete transpose L_FP = L_HJB^T
    is obtained on demand via the iterator's ``adjoint_mode="jacobian_transpose"``,
    which assembles the FP step as the transpose of the HJB Jacobian.

    Args:
        problem: MFG problem
        scheme: FDM variant (FDM_UPWIND or FDM_CENTERED)
        hjb_config: HJB solver config
        fp_config: FP solver config

    Returns:
        (HJBFDMSolver, FPFDMSolver) tuple
    """
    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

    # Map scheme to FP advection scheme
    if scheme == NumericalScheme.FDM_UPWIND:
        # FDM upwind uses divergence_upwind for FP (mass conservative, handles boundaries correctly)
        # Note: gradient_upwind has boundary flux bug, see Issue #382
        fp_config.setdefault("advection_scheme", "divergence_upwind")
    elif scheme == NumericalScheme.FDM_CENTERED:
        # FDM centered uses divergence_centered: 2nd-order central and CONSERVATIVE
        # (telescoping flux, zero boundary flux). The non-conservative gradient_centered
        # (v.grad(m)) leaks probability mass through no-flux walls (Issue #1149); it stays
        # available as an explicit advection_scheme but is no longer the centered default.
        # Both are second-order and oscillate for cell-Peclet > 2.
        fp_config.setdefault("advection_scheme", "divergence_centered")

    # Create solvers
    hjb_solver = HJBFDMSolver(problem, **hjb_config)
    fp_solver = FPFDMSolver(problem, **fp_config)

    return hjb_solver, fp_solver


def _create_sl_pair(
    problem: MFGProblem,
    scheme: NumericalScheme,
    hjb_config: dict[str, Any],
    fp_config: dict[str, Any],
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """
    Create Semi-Lagrangian HJB-FP solver pair.

    Discrete duality (Type A): Forward splatting (FP) is transpose of backward
    interpolation (HJB). Uses FPSLSolver (forward SL) for proper duality.

    Args:
        problem: MFG problem
        scheme: SL variant (SL_LINEAR or SL_CUBIC)
        hjb_config: HJB solver config
        fp_config: FP solver config

    Returns:
        (HJBSemiLagrangianSolver, FPSLSolver) tuple

    Note:
        FPSLSolver (forward SL, splatting) is the adjoint of HJB backward SL.
        FPSLJacobianSolver (backward SL with Jacobian) is deprecated.
    """
    from mfgarchon.alg.numerical.fp_solvers import FPSLSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

    # Map scheme to interpolation method
    if scheme == NumericalScheme.SL_LINEAR:
        hjb_config.setdefault("interpolation_method", "linear")
        # FP adjoint solver currently only supports linear (forward splatting)
        # Cubic interpolation in FP is future work (Issue #583)

    elif scheme == NumericalScheme.SL_CUBIC:
        hjb_config.setdefault("interpolation_method", "cubic")
        # Note: Cubic FP adjoint not yet implemented
        # For now, use linear splatting even with cubic HJB interpolation
        # This breaks exact duality but maintains O(h^2) convergence

    # Create solvers
    hjb_solver = HJBSemiLagrangianSolver(problem, **hjb_config)
    fp_solver = FPSLSolver(problem, **fp_config)

    return hjb_solver, fp_solver


def _create_gfdm_pair(
    problem: MFGProblem,
    hjb_config: dict[str, Any],
    fp_config: dict[str, Any],
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """
    Create GFDM HJB-FP solver pair.

    Continuous duality (Type B): L_FP = L_HJB^T + O(h) asymptotically.
    Requires renormalization for Nash gap convergence.

    Args:
        problem: MFG problem
        hjb_config: HJB solver config
        fp_config: FP solver config

    Returns:
        (HJBGFDMSolver, FPGFDMSolver) tuple

    Note:
        GFDM uses asymmetric neighborhoods (nearest neighbors differ for different
        points), so discrete transpose is not exact. However, the continuous
        operators are adjoints, giving O(h) error. Renormalization compensates
        for this during fixed-point iteration.
    """
    from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

    # Thread common GFDM parameters to both solvers
    # If delta specified for one, use it for both (consistency)
    if "delta" in hjb_config and "delta" not in fp_config:
        fp_config["delta"] = hjb_config["delta"]
    elif "delta" in fp_config and "delta" not in hjb_config:
        hjb_config["delta"] = fp_config["delta"]

    # Thread collocation_points if specified
    if "collocation_points" in hjb_config and "collocation_points" not in fp_config:
        fp_config["collocation_points"] = hjb_config["collocation_points"]
    elif "collocation_points" in fp_config and "collocation_points" not in hjb_config:
        hjb_config["collocation_points"] = fp_config["collocation_points"]

    # Create solvers
    hjb_solver = HJBGFDMSolver(problem, **hjb_config)
    fp_solver = FPGFDMSolver(problem, **fp_config)

    return hjb_solver, fp_solver


def _create_meshless_galerkin_pair(
    problem: MFGProblem,
    hjb_config: dict[str, Any],
    fp_config: dict[str, Any],
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """Create the meshless Galerkin (MLS) HJB-FP solver pair.

    Discrete duality (Type A): the FP operator is the exact transpose of the
    Galerkin HJB operator, so no renormalization is needed. Requires
    ``collocation_points`` (and typically ``delta``) in the configs; these are
    threaded across HJB/FP for consistency, mirroring the GFDM pair.
    """
    from mfgarchon.alg.numerical.meshless_galerkin import (
        MeshlessGalerkinFPSolver,
        MeshlessGalerkinHJBSolver,
    )

    # Thread shared keys across HJB/FP so the pair builds the SAME operators.
    # domain (#1139) and nitsche_penalty (#1138) MUST match too, or the clipped /
    # Nitsche blocks differ between HJB and FP and the Type-A transpose identity breaks.
    for key in (
        "delta",
        "collocation_points",
        "degree",
        "n_gauss",
        "backend",
        "domain",
        "nitsche_penalty",
        "streamline_diffusion_scale",
    ):
        if key in hjb_config and key not in fp_config:
            fp_config[key] = hjb_config[key]
        elif key in fp_config and key not in hjb_config:
            hjb_config[key] = fp_config[key]

    hjb_solver = MeshlessGalerkinHJBSolver(problem, **hjb_config)
    fp_solver = MeshlessGalerkinFPSolver(problem, **fp_config)
    return hjb_solver, fp_solver


def _create_fem_pair(
    problem: MFGProblem,
    scheme: NumericalScheme,
    hjb_config: dict[str, Any],
    fp_config: dict[str, Any],
) -> tuple[BaseHJBSolver, BaseFPSolver]:
    """
    Create FEM HJB-FP solver pair using scikit-fem backend.

    Discrete duality (Type A): Stiffness and mass matrices are symmetric,
    so L_FP = L_HJB^T exactly.

    Requires:
        - Unstructured mesh geometry (Mesh2D/Mesh3D with mesh_data)
        - scikit-fem installed (pip install scikit-fem)

    Args:
        problem: MFG problem with unstructured mesh geometry
        scheme: FEM_P1 or FEM_P2
        hjb_config: HJB solver config
        fp_config: FP solver config

    Returns:
        (HJBFEMSolver, FPFEMSolver) tuple
    """
    from mfgarchon.alg.numerical.fem import FPFEMSolver, HJBFEMSolver

    order = 1 if scheme == NumericalScheme.FEM_P1 else 2

    hjb_solver = HJBFEMSolver(problem, order=order, **hjb_config)
    fp_solver = FPFEMSolver(problem, order=order, **fp_config)

    return hjb_solver, fp_solver


def get_recommended_scheme(problem: MFGProblem) -> NumericalScheme:
    """
    Recommend a numerical scheme based on problem geometry (Phase 3 integration).

    This function performs intelligent scheme selection based on:
    - Grid regularity (structured → FDM, unstructured → GFDM)
    - Domain complexity (simple → FDM, complex boundaries → GFDM)
    - Dimension (1D → FDM, 2D+ → consider GFDM for flexibility)

    Args:
        problem: MFG problem to analyze

    Returns:
        Recommended NumericalScheme

    Note:
        This is a placeholder for Phase 3 auto-selection logic.
        For now, returns FDM_UPWIND as safe default.
    """
    from mfgarchon.geometry.protocol import GeometryType

    geo_type = getattr(problem.geometry, "geometry_type", None)

    if geo_type == GeometryType.UNSTRUCTURED_MESH:
        # Unstructured mesh -> FEM (Issue #773)
        return NumericalScheme.FEM_P1
    elif geo_type == GeometryType.CARTESIAN_GRID:
        return NumericalScheme.FDM_UPWIND
    else:
        # Default: FDM upwind (stable, works everywhere)
        return NumericalScheme.FDM_UPWIND
