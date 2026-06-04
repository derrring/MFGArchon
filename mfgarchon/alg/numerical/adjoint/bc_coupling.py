"""
Boundary condition coupling utilities for adjoint-consistent MFG systems.

This module provides utilities for creating state-dependent boundary conditions
where the BC for one equation depends on the solution of another equation.

Key use case: Adjoint-consistent reflecting boundaries for MFG (Issue #574)
- FP equation: Zero-flux BC (J·n = 0)
- HJB equation: Robin BC where ∂U/∂n = -σ²/2 · ∂ln(m)/∂n

This implementation uses the existing Robin BC framework (BCType.ROBIN with
alpha=0, beta=1) for dimension-agnostic support and clean integration with
all solver backends (FDM, GFDM, FEM, particle methods).

Mathematical Background:
-----------------------
At reflecting boundaries, the zero total flux condition for FP:
    J · n = 0  where  J = -σ²/2 · ∇m + m · α

At equilibrium, this gives:
    α = σ²/2 · ∇m/m = σ²/2 · ∇(ln m)

For quadratic Hamiltonian, α = -∇U, so:
    ∇U = -σ²/2 · ∇(ln m)

This creates a Robin-type BC for HJB that couples to the FP density gradient.

Architecture:
------------
Instead of manually threading bc_values through solver chains, this module:
1. Creates BoundaryConditions objects with Robin BC segments
2. Computes Robin BC values from current FP density
3. Returns BC object that integrates seamlessly with applicator framework
4. Solver applies BC using existing infrastructure (no special handling)

Location:
---------
This module is part of mfgarchon/alg/numerical/adjoint/ because discrete adjoint
consistency is specific to numerical PDE methods (FDM, SL, GFDM). Neural and RL
methods use different coupling mechanisms.

References:
-----------
- Issue #574: https://github.com/derrring/mfgarchon/issues/574
- Issue #704: https://github.com/derrring/mfgarchon/issues/704
- Protocol: docs/development/TOWEL_ON_BEACH_1D_PROTOCOL.md § BC Consistency
- Theory: Issue #625 (state-dependent BC coupling)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# Import BC types from geometry submodules directly for clarity
from mfgarchon.geometry.boundary.conditions import BoundaryConditions
from mfgarchon.geometry.boundary.types import BCSegment, BCType
from mfgarchon.utils.pde_coefficients import diffusion_from_volatility

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfgarchon.geometry.protocol import GeometryProtocol


def compute_boundary_log_density_gradient_1d(
    m: NDArray[np.floating],
    dx: float,
    side: str,
    regularization: float = 1e-10,
) -> float:
    """
    Compute ∂ln(m)/∂n at 1D boundary using one-sided finite differences.

    This is a dimension-specific implementation for 1D problems. For nD problems,
    use compute_adjoint_consistent_bc_values() which handles arbitrary dimensions.

    Args:
        m: Density array (interior points only, shape (Nx,))
        dx: Grid spacing
        side: Boundary side ("left" or "right")
        regularization: Small positive constant to prevent log(0)

    Returns:
        Outward normal derivative of ln(m) at boundary

    Note:
        Uses one-sided differences with outward normal convention:
        - Left boundary: normal points in -x direction
        - Right boundary: normal points in +x direction
    """
    # Regularize density to prevent log(0)
    m_safe = m + regularization

    # Compute log density
    ln_m = np.log(m_safe)

    if side == "left":
        # Left boundary: outward normal points in -x direction
        # Forward difference: ∂ln(m)/∂x ≈ (ln_m[1] - ln_m[0]) / dx
        # Outward derivative: ∂ln(m)/∂n = -∂ln(m)/∂x
        grad_ln_m = -(ln_m[1] - ln_m[0]) / dx
    elif side == "right":
        # Right boundary: outward normal points in +x direction
        # Backward difference: ∂ln(m)/∂x ≈ (ln_m[-1] - ln_m[-2]) / dx
        # Outward derivative: ∂ln(m)/∂n = ∂ln(m)/∂x
        grad_ln_m = (ln_m[-1] - ln_m[-2]) / dx
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side}")

    return float(grad_ln_m)


def create_adjoint_consistent_bc_1d(
    m_current: NDArray[np.floating],
    dx: float,
    sigma: float,
    domain_bounds: NDArray[np.floating] | None = None,
    regularization: float = 1e-10,
) -> BoundaryConditions:
    """
    Create adjoint-consistent Robin BC for 1D HJB equation.

    This function creates a BoundaryConditions object with Robin BC segments
    that couple the HJB boundary condition to the current FP density gradient.
    Uses the existing Robin BC framework for clean integration.

    Mathematical Formula:
        At reflecting boundaries: ∂U/∂n = -σ²/2 · ∂ln(m)/∂n

    Implementation:
        Robin BC: alpha*U + beta*∂U/∂n = g
        - alpha = 0.0 (no U term)
        - beta = 1.0 (coefficient of ∂U/∂n)
        - g = -σ²/2 · ∂ln(m)/∂n (computed from density)

    Args:
        m_current: Current FP density (interior points, shape (Nx,))
        dx: Grid spacing
        sigma: Diffusion coefficient
        domain_bounds: Domain bounds array of shape (1, 2), optional
        regularization: Regularization constant for log(m)

    Returns:
        BoundaryConditions object with Robin BC segments for both boundaries

    Example:
        >>> from mfgarchon.alg.numerical.adjoint import create_adjoint_consistent_bc_1d
        >>> # In Picard iteration
        >>> m_current = solve_fp(U_prev)
        >>> hjb_bc = create_adjoint_consistent_bc_1d(
        ...     m_current=m_current[-1, :],  # Final time slice
        ...     dx=problem.dx,
        ...     sigma=problem.sigma,
        ...     domain_bounds=problem.geometry.domain_bounds,
        ... )
        >>> U_new = hjb_solver.solve_hjb_system(bc=hjb_bc, ...)

    Note:
        This is a 1D-specific implementation. For nD problems, the architecture
        is the same but gradient computation needs dimension-agnostic handling
        via geometry.get_gradient_operator().
    """
    # Compute density gradients at boundaries
    grad_ln_m_left = compute_boundary_log_density_gradient_1d(m_current, dx, side="left", regularization=regularization)
    grad_ln_m_right = compute_boundary_log_density_gradient_1d(
        m_current, dx, side="right", regularization=regularization
    )

    # Robin BC values: g = -σ²/2 · ∂ln(m)/∂n
    diffusion_coeff = diffusion_from_volatility(sigma)
    value_left = -diffusion_coeff * grad_ln_m_left
    value_right = -diffusion_coeff * grad_ln_m_right

    # Create Robin BC segments (alpha=0, beta=1, value=g)
    segments = [
        BCSegment(
            name="left_adjoint_consistent",
            bc_type=BCType.ROBIN,
            alpha=0.0,  # No U term
            beta=1.0,  # Coefficient of ∂U/∂n
            value=value_left,  # -σ²/2 · ∂ln(m)/∂n at left boundary
            boundary="x_min",
            priority=1,
        ),
        BCSegment(
            name="right_adjoint_consistent",
            bc_type=BCType.ROBIN,
            alpha=0.0,
            beta=1.0,
            value=value_right,  # -σ²/2 · ∂ln(m)/∂n at right boundary
            boundary="x_max",
            priority=1,
        ),
    ]

    # Create BC with Robin segments (direct construction)
    return BoundaryConditions(
        segments=segments,
        dimension=1,
        domain_bounds=domain_bounds,
        default_bc=BCType.NEUMANN,  # Fallback (shouldn't be reached)
        default_value=0.0,
    )


def compute_adjoint_consistent_bc_values(
    m_current: NDArray[np.floating],
    geometry: GeometryProtocol,
    sigma: float,
    dimension: int = 1,
    regularization: float = 1e-10,
) -> BoundaryConditions:
    """
    Create adjoint-consistent Robin BC for HJB equation (dimension-agnostic).

    This is the general interface that dispatches to dimension-specific
    implementations. Currently only 1D is implemented.

    Args:
        m_current: Current FP density (interior points)
        geometry: Geometry object providing grid spacing and bounds
        sigma: Diffusion coefficient
        dimension: Spatial dimension
        regularization: Regularization constant for log(m)

    Returns:
        BoundaryConditions object with adjoint-consistent Robin BC

    Raises:
        NotImplementedError: For dimensions > 1 (planned for future releases)

    Example:
        >>> from mfgarchon.alg.numerical.adjoint import compute_adjoint_consistent_bc_values
        >>> bc = compute_adjoint_consistent_bc_values(
        ...     m_current=m[-1, :],
        ...     geometry=problem.geometry,
        ...     sigma=problem.sigma,
        ...     dimension=problem.dimension,
        ... )
        >>> U = hjb_solver.solve_hjb_system(bc=bc, ...)
    """
    if dimension == 1:
        # 1D implementation using one-sided finite differences
        dx = geometry.get_grid_spacing()[0]
        # Use getattr pattern per CLAUDE.md (no hasattr for optional attributes)
        domain_bounds = getattr(geometry, "domain_bounds", None)
        return create_adjoint_consistent_bc_1d(
            m_current=m_current,
            dx=dx,
            sigma=sigma,
            domain_bounds=domain_bounds,
            regularization=regularization,
        )
    else:
        # Placeholder for 2D/nD implementation
        # TODO: Implement using geometry.get_gradient_operator()
        raise NotImplementedError(
            f"Adjoint-consistent BC not yet implemented for {dimension}D. "
            f"Currently only 1D is supported. Extension to nD requires normal gradient "
            f"computation via geometry.get_gradient_operator() at boundary points."
        )


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Alias for old function name (from flawed implementation)
# Will be deprecated in v0.18.0
compute_coupled_hjb_bc_values = compute_adjoint_consistent_bc_values


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    """Quick validation of boundary gradient computation and Robin BC creation."""
    print("Testing adjoint-consistent BC creation...")
    print(f"Module location: {__file__}")
    print()

    # Test case 1: Exponential density (known analytical gradient)
    x = np.linspace(0, 1, 11)
    m_exp = np.exp(-x)  # m(x) = exp(-x)
    # Analytical: d/dx[ln(exp(-x))] = d/dx[-x] = -1
    # At left (x=0): ∂ln(m)/∂n = -(-1) = 1 (outward is -x direction)
    # At right (x=1): ∂ln(m)/∂n = -1 (outward is +x direction)

    dx = x[1] - x[0]
    grad_left = compute_boundary_log_density_gradient_1d(m_exp, dx, "left")
    grad_right = compute_boundary_log_density_gradient_1d(m_exp, dx, "right")

    print("Test 1: Exponential density m(x) = exp(-x)")
    print("  Analytical: dln(m)/dn|_left = 1.0, dln(m)/dn|_right = -1.0")
    print(f"  Numerical:  dln(m)/dn|_left = {grad_left:.6f}, dln(m)/dn|_right = {grad_right:.6f}")
    print(f"  Error: left = {abs(grad_left - 1.0):.2e}, right = {abs(grad_right - (-1.0)):.2e}")
    print()

    # Test case 2: Create Robin BC object
    print("Test 2: Robin BC creation")
    sigma = 0.2
    domain_bounds = np.array([[0.0, 1.0]])
    bc = create_adjoint_consistent_bc_1d(
        m_current=m_exp,
        dx=dx,
        sigma=sigma,
        domain_bounds=domain_bounds,
    )

    print("  Density: m(x) = exp(-x)")
    print(f"  Diffusion: sigma = {sigma}")
    print(f"  BC object: {bc}")
    print()

    # Verify BC properties
    assert bc.dimension == 1, "Dimension should be 1"
    assert bc.is_mixed, "Should be mixed BC (multiple segments)"
    assert len(bc.segments) == 2, "Should have 2 segments (left + right)"

    left_seg = bc.segments[0]
    right_seg = bc.segments[1]

    assert left_seg.bc_type == BCType.ROBIN, "Left should be Robin BC"
    assert right_seg.bc_type == BCType.ROBIN, "Right should be Robin BC"
    assert left_seg.alpha == 0.0, "alpha should be 0 (no U term)"
    assert left_seg.beta == 1.0, "beta should be 1 (dU/dn coefficient)"

    # Check BC values
    expected_left = -(sigma**2) / 2 * 1.0  # grad_ln_m_left ~ 1.0
    expected_right = -(sigma**2) / 2 * (-1.0)  # grad_ln_m_right ~ -1.0

    print(f"  Left BC value: {left_seg.value:.6f} (expected ~ {expected_left:.6f})")
    print(f"  Right BC value: {right_seg.value:.6f} (expected ~ {expected_right:.6f})")
    print()

    print("All tests passed! Robin BC created correctly using framework.")
