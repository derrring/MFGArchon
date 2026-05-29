"""
Numerical Scheme Enumerations for MFG Solvers

This module defines the numerical discretization schemes available in MFGarchon
for solving coupled HJB-FP systems with mathematical duality guarantees.

The NumericalScheme enum is used with the three-mode solving API:
    - Safe Mode: problem.solve(scheme=NumericalScheme.FDM_UPWIND)
    - Expert Mode: problem.solve(hjb_solver=..., fp_solver=...)
    - Auto Mode: problem.solve()  # Auto-selects based on geometry

See docs/architecture/FACTORY_PATTERN_DESIGN.md for complete design.
See Issue #706 (adjoint operators) for mathematical foundation.

Related:
    - Issue #580: Adjoint-aware solver pairing
    - SchemeFamily enum: Used internally for duality validation
"""

from __future__ import annotations

from enum import Enum


class NumericalScheme(Enum):
    """
    Numerical discretization schemes for MFG with duality guarantees.

    This enum defines the available schemes for the Safe Mode API:
        >>> from mfgarchon import create_crowd_problem, NumericalScheme
        >>> problem = create_crowd_problem(...)
        >>> result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

    Duality Guarantees
    ------------------

    **Type A (Discrete Duality)**: Exact adjoint at matrix level (L_FP = L_HJB^T)

    - **FDM_UPWIND**: First-order upwind finite differences
        - Convergence: O(h^{0.5-1}) for HJB-FP coupling
        - Stability: Excellent (most stable scheme)
        - Use case: 1D/2D structured grids, production code
        - Adjoint: div_upwind is exact transpose of grad_upwind

    - **FDM_CENTERED**: Second-order centered finite differences
        - Convergence: O(h^2)
        - Stability: Requires small CFL (smooth problems only)
        - Use case: Smooth solutions without shocks
        - Adjoint: div_centered is exact transpose of grad_centered

    - **SL_LINEAR**: Semi-Lagrangian with linear interpolation
        - Convergence: O(h^2) with Δt = O(h^{3/2})
        - Stability: Unconditionally stable
        - Use case: 2D/3D problems, better scaling than FDM
        - Adjoint: Forward splatting (scatter) is transpose of backward interpolation (gather)

    - **SL_CUBIC**: Semi-Lagrangian with cubic interpolation (⚠️ EXPERIMENTAL)
        - Convergence: O(h^4) with Δt = O(h^{3/2})
        - Stability: Generally stable but can produce NaN (Issue #583)
        - Use case: High-accuracy requirements, research
        - Adjoint: Cubic splatting is transpose of cubic interpolation
        - **Status**: Experimental - known NaN issues in some cases

    **Type A (FEM)**: Discrete duality via symmetric bilinear forms

    - **FEM_P1**: Linear Lagrange finite elements on triangular/tetrahedral meshes
        - Convergence: O(h^2) in L2 norm
        - Stability: Unconditionally stable (coercive bilinear form)
        - Use case: Unstructured meshes, complex geometry, variational problems
        - Adjoint: Stiffness and mass matrices are symmetric
        - Backend: scikit-fem assembly (Issue #773)

    - **FEM_P2**: Quadratic Lagrange finite elements
        - Convergence: O(h^3) in L2 norm
        - Stability: Same as P1
        - Use case: Higher accuracy on same mesh, curved boundaries
        - Adjoint: Same symmetry as P1

    **Type B (Continuous Duality)**: Asymptotic adjoint (L_FP = L_HJB^T + O(h))

    - **GFDM**: Generalized Finite Difference Method (meshfree)
        - Convergence: O(h^2) with careful QP tuning
        - Stability: Good with monotonicity enforcement
        - Use case: Complex geometry, obstacles, unstructured grids
        - Adjoint: Complementary upwind parameters (not exact discrete transpose)
        - **Note**: Requires mass renormalization (applied automatically)

    Scheme Selection Guide
    ----------------------

    | Problem Type | Recommended Scheme | Rationale |
    |:-------------|:-------------------|:----------|
    | 1D/2D structured | FDM_UPWIND | Most stable, proven convergence |
    | 2D smooth | FDM_CENTERED | Higher accuracy if stable |
    | 2D/3D regular grid | SL_LINEAR | Better scaling, unconditionally stable |
    | High accuracy needed | SL_CUBIC | O(h^4) if stable (check for NaN) |
    | Obstacles/complex geom | GFDM | Only scheme that handles obstacles |
    | Unstructured mesh | GFDM | Meshfree handles irregular grids |
    | FEM triangle mesh | FEM_P1 | Variational structure on unstructured mesh |
    | FEM high accuracy | FEM_P2 | Quadratic elements, O(h^3) convergence |

    Auto-Selection Logic
    --------------------

    When using Auto Mode (no scheme specified), MFGarchon selects:
    - 1D/2D structured → FDM_UPWIND
    - 2D unstructured → GFDM
    - 3D → SL_LINEAR
    - Obstacles present → GFDM

    Mathematical Foundation
    -----------------------

    **Discrete Duality (Type A)**:
        The discrete operators satisfy L_FP = (L_HJB)^T exactly at the matrix level.
        This guarantees exact discrete Nash equilibrium and energy conservation.

    **Continuous Duality (Type B)**:
        Both operators converge to correct continuous adjoints as h→0, but
        L_FP ≠ (L_HJB)^T at finite h. Requires post-hoc mass conservation fixes.

    See Issue #706 (adjoint operators) for detailed mathematical treatment
    of discrete vs continuous duality in MFG systems.

    Examples
    --------

    Safe Mode (validated pairing):
        >>> from mfgarchon import create_lq_problem, NumericalScheme
        >>> problem = create_lq_problem(dimension=1, nx=100, T=1.0)
        >>> result = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

    Comparing schemes:
        >>> for scheme in [NumericalScheme.FDM_UPWIND, NumericalScheme.SL_LINEAR]:
        ...     result = problem.solve(scheme=scheme)
        ...     print(f"{scheme.value}: {result.iterations} iterations")

    See Also
    --------
    - SchemeFamily: Internal enum for duality validation
    - create_paired_solvers(): Factory function (internal use)
    - MFGProblem.solve(): Three-mode solving API

    References
    ----------
    - Carlini & Silva (2014): Semi-Lagrangian schemes for MFG
    - Calzola et al. (2023): High-order MFG discretizations
    - Issue #580: Adjoint-aware solver pairing RFC
    """

    # Type A: Discrete duality schemes (exact adjoint)
    FDM_UPWIND = "fdm_upwind"
    FDM_CENTERED = "fdm_centered"
    SL_LINEAR = "sl_linear"
    SL_CUBIC = "sl_cubic"  # Experimental (Issue #583)

    # Type A (FEM): Discrete duality via symmetric bilinear forms (Issue #773)
    FEM_P1 = "fem_p1"
    FEM_P2 = "fem_p2"

    # Type B: Continuous duality schemes (asymptotic adjoint)
    GFDM = "gfdm"

    # Type A (meshfree): discrete duality via Galerkin weak form on MLS (Issue #1131)
    MESHLESS_GALERKIN = "meshless_galerkin"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.value

    def is_discrete_dual(self) -> bool:
        """
        Check if this scheme has discrete duality (Type A).

        Returns:
            True if L_FP = (L_HJB)^T exactly, False if only asymptotic duality.

        Example:
            >>> NumericalScheme.FDM_UPWIND.is_discrete_dual()
            True
            >>> NumericalScheme.GFDM.is_discrete_dual()
            False
        """
        return self in {
            self.FDM_UPWIND,
            self.FDM_CENTERED,
            self.SL_LINEAR,
            self.SL_CUBIC,
            self.FEM_P1,
            self.FEM_P2,
            self.MESHLESS_GALERKIN,
        }

    def requires_renormalization(self) -> bool:
        """
        Check if this scheme requires mass renormalization.

        Type B schemes (continuous duality) require post-hoc mass conservation
        fixes because L_FP ≠ (L_HJB)^T at finite h.

        Returns:
            True if scheme requires RenormalizationWrapper, False otherwise.

        Example:
            >>> NumericalScheme.GFDM.requires_renormalization()
            True
            >>> NumericalScheme.FDM_UPWIND.requires_renormalization()
            False
        """
        return not self.is_discrete_dual()

    @property
    def is_experimental(self) -> bool:
        """
        Check if this scheme is experimental/unstable.

        Experimental schemes may have known issues and are not recommended
        for production use.

        Returns:
            True if scheme is experimental, False if production-ready.

        Example:
            >>> NumericalScheme.SL_CUBIC.is_experimental
            True
            >>> NumericalScheme.FDM_UPWIND.is_experimental
            False
        """
        return self == self.SL_CUBIC  # Issue #583
