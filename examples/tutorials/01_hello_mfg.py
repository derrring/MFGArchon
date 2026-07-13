"""
Tutorial 01: Hello Mean Field Games

This is the simplest possible MFG example using the Linear-Quadratic (LQ) framework.

What you'll learn:
- How to define an MFG problem using Model, Conditions, and MFGProblem
- How to solve it using problem.solve()
- How to inspect the solution

Mathematical Problem:
    A Linear-Quadratic Mean Field Game on [0,1] with:
    - Hamiltonian: H(p, m) = (1/2)|p|^2 + coupling * m
    - Initial density: Gaussian centered at 0.5
    - Terminal cost: u_T(x) = (x - 0.5)^2
"""

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

if __name__ == "__main__":
    # ==============================================================================
    # Step 1: Define the Model (game rules)
    # ==============================================================================

    print("=" * 70)
    print("TUTORIAL 01: Hello Mean Field Games")
    print("=" * 70)
    print()

    # Physical parameters
    sigma = 0.1  # SDE volatility (controls agent randomness)
    coupling = 0.5  # Congestion cost strength

    # Hamiltonian: H(p, m) = (1/2)|p|^2 + coupling * m
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: coupling * m,
        coupling_dm=lambda m: coupling,
    )

    model = Model(hamiltonian=hamiltonian, sigma=sigma)

    print("Model created:")
    print(f"  Hamiltonian: H(p, m) = |p|^2/2 + {coupling}*m")
    print(f"  Sigma (volatility): {sigma}")
    print()

    # ==============================================================================
    # Step 2: Define the Domain (spatial arena)
    # ==============================================================================

    # A deliberately small grid keeps this "hello world" fast to run end-to-end.
    # (Finer grids are correct but slow on the default FDM solver -- the inner
    # Newton assembles its Jacobian by O(Nx^2) finite differences; tracked in #1607.)
    domain = TensorProductGrid(
        bounds=[(0.0, 1.0)],  # Domain [0, 1]
        Nx_points=[21],  # 21 grid points (20 intervals)
        boundary_conditions=no_flux_bc(dimension=1),
    )

    print("Domain created:")
    print("  Spatial: [0.0, 1.0]")
    print(f"  Points: {domain.num_spatial_points}")
    print()

    # ==============================================================================
    # Step 3: Define the Conditions (problem data)
    # ==============================================================================

    conditions = Conditions(
        u_terminal=lambda x: (x - 0.5) ** 2,  # Agents want to be at x = 0.5
        m_initial=lambda x: np.exp(-50 * (x - 0.5) ** 2),  # Start near center
        T=1.0,  # Time horizon
    )

    print("Conditions created:")
    print(f"  Time horizon: T = {conditions.T}")
    print("  Terminal cost: u_T(x) = (x - 0.5)^2")
    print("  Initial density: Gaussian at x = 0.5")
    print()

    # ==============================================================================
    # Step 4: Create and Solve the Problem
    # ==============================================================================

    problem = MFGProblem(model=model, domain=domain, conditions=conditions, Nt=20)

    print("Solving MFG system...")
    print()

    result = problem.solve(verbose=True)

    print()
    print("Solution completed!")
    print()

    # ==============================================================================
    # Step 5: Inspect the Solution
    # ==============================================================================

    print("=" * 70)
    print("SOLUTION SUMMARY")
    print("=" * 70)
    print()

    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final error: {result.max_error:.6e}")
    print()

    # Solution arrays
    print("Solution arrays:")
    print(f"  U (value function): {result.U.shape}")
    print(f"  M (density): {result.M.shape}")
    print()

    # Mass conservation check
    dx = domain.get_grid_spacing()[0]
    total_mass = np.sum(result.M[-1, :]) * dx
    print(f"Final mass (should be ~1.0): {total_mass:.6f}")
    print()

    # ==============================================================================
    # Step 6: Parameter Variation (new in v1.0)
    # ==============================================================================

    print("=" * 70)
    print("PARAMETER VARIATION")
    print("=" * 70)
    print()

    # Swap the Model component (same domain, same conditions)
    model2 = Model(hamiltonian=hamiltonian, sigma=0.3)
    problem2 = problem.with_model(model2)
    result2 = problem2.solve(verbose=False)
    print(f"sigma=0.3: converged={result2.converged}, error={result2.max_error:.2e}")

    # Swap the Conditions component (same model, same domain)
    cond2 = Conditions(
        u_terminal=lambda x: (x - 0.5) ** 2,
        m_initial=lambda x: np.exp(-50 * (x - 0.5) ** 2),
        T=2.0,
    )
    problem3 = problem.with_conditions(cond2)
    result3 = problem3.solve(verbose=False)
    print(f"T=2.0: converged={result3.converged}, error={result3.max_error:.2e}")
    print()

    # ==============================================================================
    # Step 7: Visualize (Optional)
    # ==============================================================================

    print("=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    print()

    try:
        from pathlib import Path

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        x = domain.coordinates[0]
        t = np.linspace(0, problem.T, problem.Nt + 1)

        # Plot value function at final time
        axes[0].plot(x, result.U[-1, :])
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u(T, x)")
        axes[0].set_title("Terminal Value Function")
        axes[0].grid(True, alpha=0.3)

        # Plot density evolution
        X, T_grid = np.meshgrid(x, t)
        contour = axes[1].contourf(X, T_grid, result.M, levels=20, cmap="viridis")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("t")
        axes[1].set_title("Density Evolution m(t,x)")
        plt.colorbar(contour, ax=axes[1])

        plt.tight_layout()

        output_dir = Path(__file__).parent.parent / "outputs" / "tutorials"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "01_hello_mfg.png"
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
    print("  1. Model: game rules (Hamiltonian + sigma)")
    print("  2. Domain: spatial grid (TensorProductGrid)")
    print("  3. Conditions: problem data (u_terminal, m_initial, T)")
    print("  4. MFGProblem(model, domain, conditions, Nt=20)")
    print("  5. result = problem.solve()")
    print("  6. Parameter variation: problem.with_model(), problem.with_conditions()")
    print()
    print("Next: Tutorial 02 - Custom Hamiltonian")
    print("=" * 70)
