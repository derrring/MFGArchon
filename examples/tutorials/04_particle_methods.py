"""
Tutorial 04: Advanced Solver Configuration

Learn how to configure MFG solvers with custom parameters.

What you'll learn:
- How to use MFGSolverConfig for fine-grained control
- How to compare different solver parameters
- How to tune convergence settings
- The SolverFactory API for advanced use cases

Note: Particle methods are available in the research repository (MFG-Research)
for specialized applications. This tutorial covers the standard grid-based
solver configuration available in the core package.
"""

from pathlib import Path

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.config import MFGSolverConfig, PicardConfig
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.factory import SolverFactory
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

if __name__ == "__main__":
    # ==============================================================================
    # Step 1: Default Solution
    # ==============================================================================

    print("=" * 70)
    print("TUTORIAL 04: Advanced Solver Configuration")
    print("=" * 70)
    print()

    # Create grid and components
    grid = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))

    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.3 * m,
        coupling_dm=lambda m: 0.3,
    )

    model = Model(hamiltonian=hamiltonian, sigma=0.15)

    # The caller normalises: the library measures the mass and reports it, but does not rescale your
    # density (Issue #1887). `grid.integrate` is the grid's own measure -- on this endpoint-inclusive
    # grid the two wall nodes own half a cell each, so it is not `sum(m) * dx` (Issue #2145).
    def bump(x):
        return np.exp(-50 * (x - 0.5) ** 2)

    bump_mass = float(grid.integrate(bump(np.asarray(grid.coordinates[0]))))

    conditions = Conditions(
        u_terminal=lambda x: (x - 0.5) ** 2,
        m_initial=lambda x: bump(x) / bump_mass,
        T=1.0,
    )

    # Create problem
    problem = MFGProblem(
        model=model,
        domain=grid,
        conditions=conditions,
        Nt=50,
    )

    print("Solving with DEFAULT settings...")
    print("-" * 70)

    result_default = problem.solve(verbose=True)

    print()
    print(f"Default: Converged in {result_default.iterations} iterations")
    print(f"  Final error: {result_default.max_error:.6e}")
    print()

    # ==============================================================================
    # Step 2: Custom Configuration with SolverConfig
    # ==============================================================================

    print("=" * 70)
    print("CUSTOM CONFIGURATION")
    print("=" * 70)
    print()

    # MFGSolverConfig provides fine-grained control over solver behavior
    print("Creating custom configuration...")
    print()

    # Configuration with tighter tolerance
    config_tight = MFGSolverConfig(
        picard=PicardConfig(
            max_iterations=50,  # More iterations allowed
            tolerance=1e-6,  # Tighter convergence tolerance
        )
    )

    print(f"Config: max_iterations={config_tight.picard.max_iterations}, tol={config_tight.picard.tolerance}")
    print()

    # Create solver with custom config using SolverFactory
    solver_tight = SolverFactory.create_solver(problem, config=config_tight)
    result_tight = solver_tight.solve(verbose=True)

    print()
    print(f"Tight tolerance: Converged in {result_tight.iterations} iterations")
    print(f"  Final error: {result_tight.max_error:.6e}")
    print()

    # ==============================================================================
    # Step 3: Looser Tolerance for Speed
    # ==============================================================================

    print("=" * 70)
    print("TOLERANCE VS SPEED TRADE-OFF")
    print("=" * 70)
    print()

    # Configuration with looser tolerance for faster (but less accurate) results
    config_fast = MFGSolverConfig(
        picard=PicardConfig(
            max_iterations=20,
            tolerance=1e-3,  # Looser tolerance
        )
    )

    solver_fast = SolverFactory.create_solver(problem, config=config_fast)
    result_fast = solver_fast.solve(verbose=True)

    print()
    print(f"Fast (loose tol): Converged in {result_fast.iterations} iterations")
    print(f"  Final error: {result_fast.max_error:.6e}")
    print()

    # ==============================================================================
    # Step 4: Compare Solutions
    # ==============================================================================

    print("=" * 70)
    print("SOLUTION COMPARISON")
    print("=" * 70)
    print()

    # Calculate differences between solutions
    diff_tight_default = np.linalg.norm(result_tight.M - result_default.M) / np.linalg.norm(result_default.M)
    diff_fast_default = np.linalg.norm(result_fast.M - result_default.M) / np.linalg.norm(result_default.M)

    print("Relative L2 differences from default:")
    print(f"  Tight tolerance vs Default: {diff_tight_default:.6e}")
    print(f"  Fast (loose) vs Default:    {diff_fast_default:.6e}")
    print()

    # Mass conservation check
    dx = problem.geometry.get_grid_spacing()[0]
    mass_default = np.sum(result_default.M[-1, :]) * dx
    mass_tight = np.sum(result_tight.M[-1, :]) * dx
    mass_fast = np.sum(result_fast.M[-1, :]) * dx

    print("Final mass (should be ~1.0):")
    print(f"  Default:         {mass_default:.6f}")
    print(f"  Tight tolerance: {mass_tight:.6f}")
    print(f"  Fast:            {mass_fast:.6f}")
    print()

    # ==============================================================================
    # Step 5: Configuration Summary Table
    # ==============================================================================

    print("=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print()

    configurations = [
        ("Default", result_default, 1e-5),
        ("Tight (1e-6)", result_tight, 1e-6),
        ("Fast (1e-3)", result_fast, 1e-3),
    ]

    print(f"{'Configuration':<20} {'Tolerance':<12} {'Iterations':<12} {'Final Error':<12}")
    print("-" * 70)

    for name, result, tol in configurations:
        print(f"{name:<20} {tol:<12.0e} {result.iterations:<12} {result.max_error:.4e}")

    print()

    # ==============================================================================
    # Step 6: Best Practices
    # ==============================================================================

    print("=" * 70)
    print("CONFIGURATION BEST PRACTICES")
    print("=" * 70)
    print()

    print("Tolerance Selection Guide:")
    print()
    print("  Use Case                    | Recommended Tolerance")
    print("  " + "-" * 55)
    print("  Quick prototyping           | 1e-3 to 1e-4")
    print("  Standard research           | 1e-4 to 1e-5")
    print("  Publication quality         | 1e-6 to 1e-8")
    print("  Convergence studies         | Vary systematically")
    print()

    print("Performance Tips:")
    print("  - Start with default settings, then tune as needed")
    print("  - Use loose tolerance for initial exploration")
    print("  - Tighten tolerance for final results")
    print("  - Monitor convergence history to diagnose issues")
    print()

    # ==============================================================================
    # Step 7: Visualize (Optional)
    # ==============================================================================

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: Final density comparison
        x = grid.coordinates[0]
        axes[0].plot(x, result_default.M[-1, :], "b-", linewidth=2, label="Default")
        axes[0].plot(x, result_tight.M[-1, :], "g--", linewidth=2, label="Tight", alpha=0.8)
        axes[0].plot(x, result_fast.M[-1, :], "r:", linewidth=2, label="Fast", alpha=0.8)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("m(T, x)")
        axes[0].set_title("Final Density Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Value function comparison
        axes[1].plot(x, result_default.U[-1, :], "b-", linewidth=2, label="Default")
        axes[1].plot(x, result_tight.U[-1, :], "g--", linewidth=2, label="Tight", alpha=0.8)
        axes[1].plot(x, result_fast.U[-1, :], "r:", linewidth=2, label="Fast", alpha=0.8)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("u(T, x)")
        axes[1].set_title("Terminal Value Function")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Convergence comparison (error vs iteration)
        if hasattr(result_default, "error_history_M") and result_default.error_history_M:
            axes[2].semilogy(result_default.error_history_M, "b-", label="Default", linewidth=2)
        if hasattr(result_tight, "error_history_M") and result_tight.error_history_M:
            axes[2].semilogy(result_tight.error_history_M, "g--", label="Tight", linewidth=2)
        if hasattr(result_fast, "error_history_M") and result_fast.error_history_M:
            axes[2].semilogy(result_fast.error_history_M, "r:", label="Fast", linewidth=2)
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Error (log scale)")
        axes[2].set_title("Convergence History")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        output_dir = Path(__file__).parent.parent / "outputs" / "tutorials"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "04_solver_config.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
        print()

    except ImportError:
        print("Matplotlib not available - skipping visualization")
        print()

    # ==============================================================================
    # Summary
    # ==============================================================================

    print("=" * 70)
    print("TUTORIAL COMPLETE")
    print("=" * 70)
    print()
    print("What you learned:")
    print("  1. How to use MFGSolverConfig for custom parameters")
    print("  2. How to use SolverFactory for advanced control")
    print("  3. The tolerance vs speed trade-off")
    print("  4. How to compare solutions with different settings")
    print()
    print("Key takeaway:")
    print("  Configuration choices affect both speed and accuracy.")
    print("  Start simple, then tune based on your specific needs.")
    print()
    print("Next: Tutorial 05 - Parameter Studies")
    print("=" * 70)
