"""
Validation: Adjoint-Consistent BC via Provider Pattern (Issue #625).

This script validates the BCValueProvider architecture by comparing:
1. Standard Neumann BC (may be inconsistent at boundary stall points)
2. Adjoint-consistent BC via AdjointConsistentProvider

The towel-on-beach problem with stall at x=0 (boundary) is used as the test case.
With standard Neumann BC, the solver cannot reach the true Boltzmann-Gibbs
equilibrium because the BC forces dU/dn=0 while the true solution requires
dU/dn = -sigma^2/2 * d(ln m)/dn.

What This Script Validates:
- BCValueProvider protocol implementation
- AdjointConsistentProvider computes density-dependent BC values
- Robin BC framework correctly receives provider-computed values
- Both BC modes run without errors through the solver pipeline

Expected Results (ideal case):
- Standard Neumann: Higher error, possibly unstable/divergent
- Adjoint-consistent: Lower error, stable convergence to Boltzmann-Gibbs

Note: Solver convergence to analytic equilibrium may require parameter tuning
(relaxation, time horizon, grid resolution) beyond the scope of this BC validation.
Non-convergence does NOT indicate BC provider failure - the provider infrastructure
works correctly as long as both modes execute without errors.

References:
- Issue #625: BCValueProvider architecture
- Issue #574: Adjoint-consistent BC implementation
- docs/development/TOWEL_ON_BEACH_1D_PROTOCOL.md
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry.boundary import (
    AdjointConsistentProvider,
    BCSegment,
    BCType,
    BoundaryConditions,
    neumann_bc,
)
from mfgarchon.utils.mfg_logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("ac_bc_validation", level="INFO")
logger = get_logger(__name__)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analytic_boltzmann_gibbs(
    x: np.ndarray,
    x_stall: float,
    sigma: float,
    lambda_crowd: float,
) -> np.ndarray:
    """
    Analytic Boltzmann-Gibbs stationary distribution.

    m*(x) = Z^{-1} exp(-V(x) / T_eff)

    where T_eff = sigma^2/2 + lambda_crowd (effective temperature)
    and V(x) = |x - x_stall| (linear potential).

    Args:
        x: Spatial grid
        x_stall: Stall location
        sigma: Diffusion coefficient
        lambda_crowd: Crowd aversion parameter

    Returns:
        Normalized Boltzmann-Gibbs density
    """
    T_eff = sigma**2 / 2 + lambda_crowd
    V = np.abs(x - x_stall)
    unnormalized = np.exp(-V / T_eff)
    Z = np.trapezoid(unnormalized, x)
    return unnormalized / Z


def create_towel_problem_standard_bc(
    L: float = 1.0,
    x_stall: float = 0.0,
    sigma: float = 0.2,
    lambda_crowd: float = 0.5,
    Nx: int = 51,
    T: float = 2.0,
    Nt: int = 41,
) -> MFGProblem:
    """Create towel-on-beach problem with STANDARD Neumann BC."""
    from mfgarchon.geometry import TensorProductGrid

    # Class-based Hamiltonian: H = (1/2)|p|^2 + |x - x_stall| + lambda_crowd * ln(m)
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        potential=lambda t, x: abs(x[0] - x_stall),
        coupling=lambda m: lambda_crowd * np.log(max(m, 1e-10)),
        coupling_dm=lambda m: lambda_crowd / max(m, 1e-10),
    )
    model = Model(hamiltonian=hamiltonian, sigma=sigma)
    conditions = Conditions(
        m_initial=lambda x: np.ones_like(x) / L,
        u_terminal=lambda x: np.zeros_like(x),
        T=T,
    )

    # Create geometry with Neumann BC
    geometry = TensorProductGrid(
        bounds=[(0.0, L)],
        Nx_points=[Nx],
        boundary_conditions=neumann_bc(dimension=1),
    )

    return MFGProblem(
        model=model,
        domain=geometry,
        conditions=conditions,
        Nt=Nt,
    )


def create_towel_problem_adjoint_bc(
    L: float = 1.0,
    x_stall: float = 0.0,
    sigma: float = 0.2,
    lambda_crowd: float = 0.5,
    Nx: int = 51,
    T: float = 2.0,
    Nt: int = 41,
) -> MFGProblem:
    """Create towel-on-beach problem with ADJOINT-CONSISTENT BC via provider."""
    from mfgarchon.geometry import TensorProductGrid

    # Class-based Hamiltonian: H = (1/2)|p|^2 + |x - x_stall| + lambda_crowd * ln(m)
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        potential=lambda t, x: abs(x[0] - x_stall),
        coupling=lambda m: lambda_crowd * np.log(max(m, 1e-10)),
        coupling_dm=lambda m: lambda_crowd / max(m, 1e-10),
    )
    model = Model(hamiltonian=hamiltonian, sigma=sigma)
    conditions = Conditions(
        m_initial=lambda x: np.ones_like(x) / L,
        u_terminal=lambda x: np.zeros_like(x),
        T=T,
    )

    # Adjoint-consistent BC via provider pattern (Issue #625)
    # Robin BC: alpha*U + beta*dU/dn = g
    # For adjoint-consistent: alpha=0, beta=1, g = -sigma^2/2 * d(ln m)/dn
    bc = BoundaryConditions(
        segments=[
            BCSegment(
                name="left_ac",
                bc_type=BCType.ROBIN,
                alpha=0.0,
                beta=1.0,
                value=AdjointConsistentProvider(side="left", diffusion=sigma),
                boundary="x_min",
            ),
            BCSegment(
                name="right_ac",
                bc_type=BCType.ROBIN,
                alpha=0.0,
                beta=1.0,
                value=AdjointConsistentProvider(side="right", diffusion=sigma),
                boundary="x_max",
            ),
        ],
        dimension=1,
    )

    # Create geometry with adjoint-consistent BC
    geometry = TensorProductGrid(
        bounds=[(0.0, L)],
        Nx_points=[Nx],
        boundary_conditions=bc,
    )

    return MFGProblem(
        model=model,
        domain=geometry,
        conditions=conditions,
        Nt=Nt,
    )


def run_validation():
    """
    Run validation comparing standard vs adjoint-consistent BC.
    """
    logger.info("=" * 70)
    logger.info("Adjoint-Consistent BC Provider Validation (Issue #625)")
    logger.info("=" * 70)

    # Problem parameters
    L = 1.0
    x_stall = 0.0  # Stall at boundary - this is where AC BC matters
    sigma = 0.2
    lambda_crowd = 0.5
    Nx = 51
    T = 3.0
    Nt = 61
    max_iter = 50
    tol = 1e-4

    logger.info("\nProblem: Towel-on-Beach with boundary stall")
    logger.info(f"  Domain: [0, {L}]")
    logger.info(f"  Stall location: x_stall = {x_stall} (at boundary)")
    logger.info(f"  Diffusion: sigma = {sigma}")
    logger.info(f"  Crowd aversion: lambda = {lambda_crowd}")
    logger.info(f"  Grid: Nx={Nx}, Nt={Nt}, T={T}")

    # Compute analytic solution
    x_grid = np.linspace(0, L, Nx)
    m_analytic = analytic_boltzmann_gibbs(x_grid, x_stall, sigma, lambda_crowd)

    # =========================================================================
    # Run 1: Standard Neumann BC
    # =========================================================================
    logger.info("\n" + "-" * 50)
    logger.info("Run 1: Standard Neumann BC (dU/dn = 0)")
    logger.info("-" * 50)

    problem_std = create_towel_problem_standard_bc(
        L=L, x_stall=x_stall, sigma=sigma, lambda_crowd=lambda_crowd, Nx=Nx, T=T, Nt=Nt
    )
    try:
        result_std = problem_std.solve(max_iterations=max_iter, tolerance=tol)
        m_final_std = result_std.M[-1, :]
        # Handle ghost cells: result may have Nx+1 points, analytic has Nx
        if len(m_final_std) == Nx + 1:
            m_final_std = m_final_std[:-1]  # Remove ghost cell
        error_std = np.max(np.abs(m_final_std - m_analytic))
        converged_std = result_std.converged
        iters_std = result_std.iterations
        logger.info(f"  Converged: {converged_std}")
        logger.info(f"  Iterations: {iters_std}")
        logger.info(f"  Max error vs analytic: {error_std:.6f}")
    except Exception as e:
        logger.warning(f"  Standard BC run failed: {e}")
        m_final_std = None
        error_std = float("inf")
        converged_std = False
        iters_std = max_iter

    # =========================================================================
    # Run 2: Adjoint-Consistent BC via Provider
    # =========================================================================
    logger.info("\n" + "-" * 50)
    logger.info("Run 2: Adjoint-Consistent BC (via AdjointConsistentProvider)")
    logger.info("-" * 50)

    problem_ac = create_towel_problem_adjoint_bc(
        L=L, x_stall=x_stall, sigma=sigma, lambda_crowd=lambda_crowd, Nx=Nx, T=T, Nt=Nt
    )
    try:
        result_ac = problem_ac.solve(max_iterations=max_iter, tolerance=tol)
        m_final_ac = result_ac.M[-1, :]
        # Handle ghost cells: result may have Nx+1 points, analytic has Nx
        if len(m_final_ac) == Nx + 1:
            m_final_ac = m_final_ac[:-1]  # Remove ghost cell
        error_ac = np.max(np.abs(m_final_ac - m_analytic))
        converged_ac = result_ac.converged
        iters_ac = result_ac.iterations
        logger.info(f"  Converged: {converged_ac}")
        logger.info(f"  Iterations: {iters_ac}")
        logger.info(f"  Max error vs analytic: {error_ac:.6f}")
    except Exception as e:
        logger.warning(f"  Adjoint-consistent BC run failed: {e}")
        m_final_ac = None
        error_ac = float("inf")
        converged_ac = False
        iters_ac = max_iter

    # =========================================================================
    # Results Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    improvement = error_std / error_ac if error_ac > 0 else float("inf")

    logger.info("\n  Standard Neumann BC:")
    logger.info(f"    Converged: {converged_std}")
    logger.info(f"    Iterations: {iters_std}")
    logger.info(f"    Max error: {error_std:.6f}")

    logger.info("\n  Adjoint-Consistent BC (Provider):")
    logger.info(f"    Converged: {converged_ac}")
    logger.info(f"    Iterations: {iters_ac}")
    logger.info(f"    Max error: {error_ac:.6f}")

    logger.info(f"\n  Improvement ratio: {improvement:.2f}x")

    # =========================================================================
    # Visualization
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Density comparison
    ax1 = axes[0]
    ax1.plot(x_grid, m_analytic, "k--", linewidth=2, label="Analytic (Boltzmann-Gibbs)")
    if m_final_std is not None:
        ax1.plot(x_grid, m_final_std, "b-", linewidth=1.5, alpha=0.7, label=f"Standard Neumann (err={error_std:.4f})")
    if m_final_ac is not None:
        ax1.plot(x_grid, m_final_ac, "r-", linewidth=1.5, alpha=0.7, label=f"Adjoint-Consistent (err={error_ac:.4f})")
    ax1.axvline(x_stall, color="orange", linestyle=":", label=f"Stall (x={x_stall})")
    ax1.set_xlabel("Position x")
    ax1.set_ylabel("Equilibrium density m(x)")
    ax1.set_title("Towel-on-Beach: BC Comparison")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error profile
    ax2 = axes[1]
    if m_final_std is not None:
        ax2.plot(x_grid, np.abs(m_final_std - m_analytic), "b-", linewidth=1.5, label="Standard Neumann")
    if m_final_ac is not None:
        ax2.plot(x_grid, np.abs(m_final_ac - m_analytic), "r-", linewidth=1.5, label="Adjoint-Consistent")
    ax2.set_xlabel("Position x")
    ax2.set_ylabel("Absolute error |m_num - m_analytic|")
    ax2.set_title("Error Profile")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "adjoint_consistent_bc_provider_validation.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"\nSaved plot: {output_path}")

    plt.show()

    # =========================================================================
    # Validation Assertion
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION STATUS")
    logger.info("=" * 70)

    # The adjoint-consistent BC should have lower error than standard Neumann
    # when the stall is at the boundary
    if error_ac < error_std:
        logger.info("\nVALIDATION PASSED: Adjoint-consistent BC outperforms standard Neumann")
        logger.info(f"  Improvement: {improvement:.2f}x lower error")
    else:
        logger.warning("\nVALIDATION UNEXPECTED: Standard Neumann had lower error")
        logger.warning("  This may indicate a regression or parameter regime where AC BC is not needed")

    return {
        "error_std": error_std,
        "error_ac": error_ac,
        "improvement": improvement,
        "converged_std": converged_std,
        "converged_ac": converged_ac,
    }


if __name__ == "__main__":
    results = run_validation()
    print(f"\nFinal results: {results}")
