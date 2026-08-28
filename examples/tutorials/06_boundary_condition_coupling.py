"""
Tutorial: Adjoint-Consistent Boundary Conditions for MFG Systems

This tutorial demonstrates how to use adjoint-consistent boundary conditions
to improve equilibrium consistency when stall points occur at domain boundaries.

Topics covered:
- Boundary condition consistency in coupled HJB-FP systems
- The BCValueProvider pattern (AdjointConsistentProvider)
- When to use adjoint-consistent vs standard BC
- Convergence improvement demonstration

Mathematical Background:
-----------------------
At reflecting boundaries with Neumann BC, the standard condition dU/dn = 0
may be inconsistent with the equilibrium (Boltzmann-Gibbs) solution when the
stall point occurs at the boundary.

The adjoint-consistent BC couples HJB to the FP density gradient:
    dU/dn = -sigma^2/2 * d(ln m)/dn

This maintains the zero-flux condition J*n = 0 at equilibrium.

Implementation:
--------------
Uses the BCValueProvider pattern (Issue #625): AdjointConsistentProvider is
stored as intent in BCSegment.value. The FixedPointIterator resolves it each
Picard iteration via problem.using_resolved_bc(state), computing concrete
Robin BC values from the current density. The solver stays generic.

Reference: docs/development/TOWEL_ON_BEACH_1D_PROTOCOL.md
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling.fixed_point_iterator import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    AdjointConsistentProvider,
    BCSegment,
    BCType,
    BoundaryConditions,
    neumann_bc,
    no_flux_bc,
)

# Problem parameters (shared by both configurations)
NX = 50
NT = 50
T = 1.0
SIGMA = 0.2
MAX_ITERATIONS = 30
TOLERANCE = 1e-6


def create_lq_model_and_conditions() -> tuple[Model, Conditions]:
    """Create standard LQ-MFG model and conditions for boundary stall scenario.

    Returns:
        (Model, Conditions) tuple for boundary stall MFG problem.
    """
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.5 * m,
        coupling_dm=lambda m: 0.5,
    )
    model = Model(hamiltonian=hamiltonian, sigma=SIGMA)
    # The caller normalises (Issue #1887): the library reports the initial mass and does not rescale
    # it. The scale is computed on a grid with this file's NX, which BOTH problem builders below use
    # -- they differ in boundary condition, not in resolution. That is worth noticing: a Conditions
    # is meant to be resolution-independent, and a scale computed on one grid is the one place where
    # that independence is only approximate. Refine NX and the discrete mass moves off 1 by the
    # quadrature error, converging to the continuum integral rather than staying pinned.
    scale_grid = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[NX + 1],
        boundary_conditions=no_flux_bc(dimension=1),
    )

    def bump(x):
        return np.exp(-20 * (x - 0.3) ** 2)

    bump_mass = float(scale_grid.integrate(bump(np.asarray(scale_grid.coordinates[0]))))

    conditions = Conditions(
        # Stall point at x=0 (boundary) - agents want to be at left edge
        m_initial=lambda x: bump(x) / bump_mass,
        u_terminal=lambda x: x**2,  # Minimal cost at x=0
        T=T,
    )
    return model, conditions


def create_standard_problem() -> MFGProblem:
    """
    Create 1D MFG problem with standard Neumann BC.

    Standard Neumann BC (dU/dn = 0) is the classical choice for reflecting
    boundaries. It works well when stall points are in the domain interior.

    Returns:
        Configured MFGProblem with Neumann BC
    """
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[NX + 1],
        boundary_conditions=neumann_bc(dimension=1),
    )
    model, conditions = create_lq_model_and_conditions()

    return MFGProblem(
        model=model,
        domain=grid,
        conditions=conditions,
        Nt=NT,
    )


def create_adjoint_consistent_problem() -> MFGProblem:
    """
    Create 1D MFG problem with adjoint-consistent BC via provider pattern.

    The AdjointConsistentProvider computes density-dependent Robin BC values:
        dU/dn = -sigma^2/2 * d(ln m)/dn

    This is stored as intent in BCSegment.value. The FixedPointIterator
    resolves it each Picard iteration using problem.using_resolved_bc(state).

    Returns:
        Configured MFGProblem with adjoint-consistent Robin BC
    """
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
                value=AdjointConsistentProvider(side="left", diffusion=SIGMA),
                boundary="x_min",
            ),
            BCSegment(
                name="right_ac",
                bc_type=BCType.ROBIN,
                alpha=0.0,
                beta=1.0,
                value=AdjointConsistentProvider(side="right", diffusion=SIGMA),
                boundary="x_max",
            ),
        ],
        dimension=1,
    )

    grid = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[NX + 1],
        boundary_conditions=bc,
    )
    model, conditions = create_lq_model_and_conditions()

    return MFGProblem(
        model=model,
        domain=grid,
        conditions=conditions,
        Nt=NT,
    )


def solve_standard(problem: MFGProblem):
    """Solve MFG with standard Neumann BC."""
    print("\nSolving with STANDARD Neumann BC (dU/dn = 0)...")

    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    iterator = FixedPointIterator(
        problem=problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
    )

    result = iterator.solve(
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
        verbose=True,
    )

    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final error (U): {result.final_error_U:.6e}")
    print(f"  Final error (M): {result.final_error_M:.6e}")

    return result


def solve_adjoint_consistent(problem: MFGProblem):
    """
    Solve MFG with adjoint-consistent BC via provider pattern.

    The BC resolution happens automatically: FixedPointIterator detects
    providers in BoundaryConditions and calls problem.using_resolved_bc()
    each iteration. The HJB solver receives concrete Robin BC values.
    """
    print("\nSolving with ADJOINT-CONSISTENT BC (via AdjointConsistentProvider)...")

    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    iterator = FixedPointIterator(
        problem=problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
    )

    result = iterator.solve(
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
        verbose=True,
    )

    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final error (U): {result.final_error_U:.6e}")
    print(f"  Final error (M): {result.final_error_M:.6e}")

    return result


def compare_bc_modes():
    """
    Main tutorial: Compare standard vs adjoint-consistent BC.

    Demonstrates:
    1. Problem setup with boundary stall
    2. Solving with both BC configurations
    3. Convergence comparison
    4. Visualization of differences
    """
    print("=" * 70)
    print("Tutorial: Adjoint-Consistent Boundary Conditions")
    print("=" * 70)

    # Step 1: Create problems
    print("\n[Step 1] Creating problems...")
    problem_std = create_standard_problem()
    problem_ac = create_adjoint_consistent_problem()

    print("  Domain: [0, 1]")
    print(f"  Grid points: {problem_std.geometry.get_grid_shape()[0]}")
    print(f"  Time steps: {problem_std.Nt}")
    print(f"  Diffusion: sigma = {SIGMA}")

    # Step 2: Solve with standard BC
    print("\n[Step 2] Standard Neumann BC (classical dU/dn = 0)...")
    result_std = solve_standard(problem_std)

    # Step 3: Solve with adjoint-consistent BC
    print("\n[Step 3] Adjoint-consistent BC (couples to d(ln m)/dn)...")
    result_ac = solve_adjoint_consistent(problem_ac)

    # Step 4: Compare results
    print("\n[Step 4] Comparison")
    print("=" * 70)

    print("\nConvergence Comparison:")
    print("  Standard BC:")
    print(f"    Iterations: {result_std.iterations}")
    print(f"    Max error: {result_std.max_error:.6e}")

    print("  Adjoint-Consistent BC:")
    print(f"    Iterations: {result_ac.iterations}")
    print(f"    Max error: {result_ac.max_error:.6e}")

    if result_ac.max_error > 0:
        improvement = result_std.max_error / result_ac.max_error
        print(f"\n  Convergence improvement: {improvement:.2f}x")

    # Solution differences
    U_diff = np.linalg.norm(result_std.U - result_ac.U)
    M_diff = np.linalg.norm(result_std.M - result_ac.M)

    print("\nSolution Differences:")
    print(f"  ||U_standard - U_ac||: {U_diff:.6e}")
    print(f"  ||M_standard - M_ac||: {M_diff:.6e}")

    # Step 5: Visualize
    print("\n[Step 5] Generating comparison plots...")
    visualize_comparison(problem_std, result_std, result_ac)

    # Summary
    print("\n" + "=" * 70)
    print("TUTORIAL SUMMARY")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Standard Neumann BC (dU/dn = 0) may be inconsistent at boundaries")
    print("  2. Adjoint-consistent BC couples to density gradient via Robin BC")
    print("  3. Use AdjointConsistentProvider in BCSegment.value for boundary stall")
    print("  4. FixedPointIterator resolves providers automatically each iteration")

    print("\nWhen to use adjoint-consistent BC:")
    print("  - Stall point at domain boundary")
    print("  - Reflecting boundaries (Neumann/no-flux)")
    print("  - High accuracy requirements")
    print("  Not needed for interior stall or periodic BC")

    plt.show()


def visualize_comparison(problem, result_std, result_ac):
    """
    Create comparison plots for standard vs adjoint-consistent BC.

    Args:
        problem: MFG problem instance (for grid info)
        result_std: Result with standard BC
        result_ac: Result with adjoint-consistent BC
    """
    # Get grid
    x = np.linspace(0, 1, problem.geometry.get_grid_shape()[0])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Comparison: Standard vs Adjoint-Consistent BC", fontsize=14, fontweight="bold")

    # Plot 1: Final density comparison
    ax = axes[0, 0]
    ax.plot(x, result_std.M[-1, :], "b-", label="Standard BC", linewidth=2)
    ax.plot(x, result_ac.M[-1, :], "r--", label="Adjoint-Consistent BC", linewidth=2)
    ax.set_xlabel("Position x")
    ax.set_ylabel("Density m(T, x)")
    ax.set_title("Final Density Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Final value function comparison
    ax = axes[0, 1]
    ax.plot(x, result_std.U[-1, :], "b-", label="Standard BC", linewidth=2)
    ax.plot(x, result_ac.U[-1, :], "r--", label="Adjoint-Consistent BC", linewidth=2)
    ax.set_xlabel("Position x")
    ax.set_ylabel("Value U(T, x)")
    ax.set_title("Final Value Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Density difference
    ax = axes[1, 0]
    m_diff = result_std.M[-1, :] - result_ac.M[-1, :]
    ax.plot(x, m_diff, "g-", linewidth=2)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Position x")
    ax.set_ylabel("m_std - m_ac")
    ax.set_title("Density Difference")
    ax.grid(True, alpha=0.3)

    # Plot 4: Convergence history
    ax = axes[1, 1]
    if len(result_std.error_history_U) > 0:
        ax.semilogy(result_std.error_history_U, "b-o", label="Standard BC", markersize=4)
        ax.semilogy(result_ac.error_history_U, "r-s", label="Adjoint-Consistent BC", markersize=4)
        ax.set_xlabel("Picard Iteration")
        ax.set_ylabel("Error ||U - U_prev||")
        ax.set_title("Convergence History")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "Convergence history not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    plt.tight_layout()


if __name__ == "__main__":
    """Run the tutorial."""
    compare_bc_modes()
