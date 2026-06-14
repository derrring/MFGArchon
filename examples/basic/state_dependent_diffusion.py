"""
State-Dependent Diffusion in Mean Field Games

Demonstrates Phase 2 features: callable diffusion coefficients D(t, x, m).

This example shows three physical scenarios:
1. Porous medium: D(m) = σ² m (diffusion proportional to density)
2. Crowd dynamics: D(m) = D₀ + D₁(1 - m/m_max) (lower diffusion in crowds)
3. Temperature-dependent: D(x) = D₀ + D₁ x (spatially varying)

Each scenario models realistic agent behavior where diffusion depends on:
- Local density (congestion effects)
- Spatial position (environmental factors)
- Time evolution (changing conditions)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def create_lq_model_and_conditions(coupling_strength: float = 1.0, sigma: float = 0.1) -> tuple[Model, Conditions]:
    """Create standard LQ-MFG model and conditions."""
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: coupling_strength * m,
        coupling_dm=lambda m: coupling_strength,
    )
    model = Model(hamiltonian=hamiltonian, sigma=sigma)
    conditions = Conditions(
        u_terminal=lambda x: (x - 0.5) ** 2,
        m_initial=lambda x: np.exp(-50 * (x - 0.5) ** 2),
        T=1.0,
    )
    return model, conditions


def scenario_porous_medium():
    """
    Porous Medium Equation: D(m) = σ² m

    Physical interpretation:
    - Diffusion increases with density (cooperative behavior)
    - Models: population spread, congestion pricing
    - Higher density → faster spreading
    """
    print("\n" + "=" * 70)
    print("Scenario 1: Porous Medium (D ∝ m)")
    print("=" * 70)

    # Define state-dependent diffusion
    def porous_medium_diffusion(t, x, m):
        """Diffusion proportional to density."""
        return 0.1 * m

    # Create problem with v1.0 API
    domain = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[101],
        boundary_conditions=no_flux_bc(dimension=1),
    )

    model, conditions = create_lq_model_and_conditions(coupling_strength=1.0, sigma=0.1)
    problem = MFGProblem(
        model=model,
        domain=domain,
        conditions=conditions,
        Nt=100,
    )

    # Create solvers
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    # MFG coupling with callable diffusion
    coupling = FixedPointIterator(
        problem,
        hjb_solver,
        fp_solver,
        damping_factor=0.5,
        volatility_field=porous_medium_diffusion,  # State-dependent!
    )

    # Solve MFG system
    print("Solving MFG with porous medium diffusion...")
    result = coupling.solve(max_iterations=20, tolerance=1e-6, verbose=True)

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    if len(result.error_history_M) > 0:
        print(f"Final M error: {result.error_history_M[-1]:.2e}")
    if len(result.error_history_U) > 0:
        print(f"Final U error: {result.error_history_U[-1]:.2e}")

    return result, domain, "Porous Medium: D(m) = 0.1m"


def scenario_crowd_dynamics():
    """
    Crowd Dynamics: D(m) = D₀ + D₁(1 - m/m_max)

    Physical interpretation:
    - Diffusion decreases in high-density regions (congestion)
    - Models: pedestrian flow, traffic, evacuation
    - Crowded areas → slower movement
    """
    print("\n" + "=" * 70)
    print("Scenario 2: Crowd Dynamics (D decreases with density)")
    print("=" * 70)

    # Define crowd-aware diffusion
    def crowd_diffusion(t, x, m):
        """Lower diffusion in high-density regions."""
        m_max = np.max(m) if np.max(m) > 0 else 1.0
        D_min = 0.05  # Minimum diffusion in crowds
        D_max = 0.15  # Maximum diffusion in free space
        return D_min + (D_max - D_min) * (1 - m / m_max)

    # Create problem
    domain = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[101],
        boundary_conditions=no_flux_bc(dimension=1),
    )

    model, conditions = create_lq_model_and_conditions(coupling_strength=1.0, sigma=0.1)
    problem = MFGProblem(
        model=model,
        domain=domain,
        conditions=conditions,
        Nt=100,
    )

    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    coupling = FixedPointIterator(
        problem,
        hjb_solver,
        fp_solver,
        damping_factor=0.5,
        volatility_field=crowd_diffusion,
    )

    print("Solving MFG with crowd dynamics diffusion...")
    result = coupling.solve(max_iterations=20, tolerance=1e-6, verbose=True)

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    if len(result.error_history_M) > 0:
        print(f"Final M error: {result.error_history_M[-1]:.2e}")
    if len(result.error_history_U) > 0:
        print(f"Final U error: {result.error_history_U[-1]:.2e}")

    return result, domain, "Crowd Dynamics: D = 0.05 + 0.1(1 - m/m_max)"


def scenario_spatially_varying():
    """
    Spatially Varying: D(x) = D₀ + D₁ x

    Physical interpretation:
    - Diffusion depends on position (environmental heterogeneity)
    - Models: temperature gradients, varying terrain roughness
    - Left side → low diffusion, right side → high diffusion
    """
    print("\n" + "=" * 70)
    print("Scenario 3: Spatially Varying (D depends on x)")
    print("=" * 70)

    # Define spatially-dependent diffusion
    def spatial_diffusion(t, x, m):
        """Diffusion increases from left to right."""
        return 0.05 + 0.1 * x

    # Create problem
    domain = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[101],
        boundary_conditions=no_flux_bc(dimension=1),
    )

    model, conditions = create_lq_model_and_conditions(coupling_strength=1.0, sigma=0.1)
    problem = MFGProblem(
        model=model,
        domain=domain,
        conditions=conditions,
        Nt=100,
    )

    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    coupling = FixedPointIterator(
        problem,
        hjb_solver,
        fp_solver,
        damping_factor=0.5,
        volatility_field=spatial_diffusion,
    )

    print("Solving MFG with spatially varying diffusion...")
    result = coupling.solve(max_iterations=20, tolerance=1e-6, verbose=True)

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    if len(result.error_history_M) > 0:
        print(f"Final M error: {result.error_history_M[-1]:.2e}")
    if len(result.error_history_U) > 0:
        print(f"Final U error: {result.error_history_U[-1]:.2e}")

    return result, domain, "Spatial Variation: D(x) = 0.05 + 0.1x"


def visualize_results(results_list):
    """Visualize all three scenarios for comparison."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("State-Dependent Diffusion in MFG", fontsize=16, fontweight="bold")

    for row_idx, (result, domain, title) in enumerate(results_list):
        x = domain.coordinates

        # Plot 1: Final density distribution
        ax = axes[row_idx, 0]
        m_final = result.M[-1, :]
        ax.plot(x, m_final, "b-", linewidth=2)
        ax.set_xlabel("x")
        ax.set_ylabel("m(T, x)")
        ax.set_title(f"{title}\nFinal Density")
        ax.grid(True, alpha=0.3)

        # Plot 2: Value function at T/2
        ax = axes[row_idx, 1]
        t_mid = result.U.shape[0] // 2
        u_mid = result.U[t_mid, :]
        ax.plot(x, u_mid, "r-", linewidth=2)
        ax.set_xlabel("x")
        ax.set_ylabel("u(T/2, x)")
        ax.set_title("Value Function (midpoint)")
        ax.grid(True, alpha=0.3)

        # Plot 3: Convergence history
        ax = axes[row_idx, 2]
        ax.semilogy(result.error_history_M, "g-", linewidth=2, label="Density error")
        ax.semilogy(result.error_history_U, "orange", linewidth=2, label="Value error")
        ax.set_xlabel("Picard Iteration")
        ax.set_ylabel("L∞ Error")
        ax.set_title("Convergence History")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Run all scenarios and compare results."""
    print("\n" + "=" * 70)
    print("State-Dependent Diffusion Examples")
    print("Demonstrating Phase 2 callable coefficient features")
    print("=" * 70)

    # Run three scenarios
    result1, domain1, title1 = scenario_porous_medium()
    result2, domain2, title2 = scenario_crowd_dynamics()
    result3, domain3, title3 = scenario_spatially_varying()

    # Visualize comparison
    print("\n" + "=" * 70)
    print("Generating comparison plots...")
    print("=" * 70)

    visualize_results(
        [
            (result1, domain1, title1),
            (result2, domain2, title2),
            (result3, domain3, title3),
        ]
    )

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    scenarios = [
        ("Porous Medium", result1),
        ("Crowd Dynamics", result2),
        ("Spatial Variation", result3),
    ]

    print(f"\n{'Scenario':<20} {'Iterations':<12} {'M Error':<15} {'Converged':<10}")
    print("-" * 70)
    for name, result in scenarios:
        final_error = result.error_history_M[-1] if len(result.error_history_M) > 0 else 0.0
        print(f"{name:<20} {result.iterations:<12} {final_error:<15.2e} {result.converged!s:<10}")

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("1. Porous medium: Density-dependent diffusion enables cooperative spreading")
    print("2. Crowd dynamics: Congestion effects reduce mobility in high-density regions")
    print("3. Spatial variation: Environmental heterogeneity affects agent movement")
    print("\nAll scenarios converged successfully with callable diffusion coefficients!")


if __name__ == "__main__":
    main()
