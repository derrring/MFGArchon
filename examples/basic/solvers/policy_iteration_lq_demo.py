"""
Demonstration of policy iteration for LQ-MFG problems.

This example shows how to apply Howard's policy iteration algorithm to solve
the Hamilton-Jacobi-Bellman equation in a Linear-Quadratic Mean Field Game.

Policy iteration alternates between:
1. Policy evaluation: Solve linearized HJB with fixed control
2. Policy improvement: Update control by maximizing Hamiltonian

For LQ problems, this typically converges faster than value iteration (fixed-point).

Theoretical Background:
======================

For a control problem with Hamiltonian:
    H(x, p, α, m) = 0.5 * |α|² + α·p + V(x, m)

Where α is the control, policy iteration performs:

1. Policy Evaluation (solve linear PDE):
   ∂u/∂t + 0.5*α_k² + α_k·∂u/∂x + V(x,m) = 0

2. Policy Improvement:
   α_{k+1}(x,t) = argmax_α [α·∂u/∂x - 0.5*α²]
                = -∂u/∂x

Convergence:
- Superlinear convergence (better than linear fixed-point)
- Typically requires fewer iterations than value iteration
- Each iteration solves a LINEAR PDE (cheaper than nonlinear)

Usage:
    python examples/basic/policy_iteration_lq_demo.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.numerical import create_lq_policy_problem


def demonstrate_policy_iteration_concept():
    """
    Demonstrate the concept of policy iteration with a simple 1D example.

    Note: This is a conceptual demonstration. Full implementation of policy iteration
    for HJB-MFG requires solving the linearized HJB equation with fixed policy,
    which is beyond the scope of this basic example.
    """
    print("=" * 80)
    print("Policy Iteration for LQ-MFG: Conceptual Demonstration")
    print("=" * 80)
    print()
    print("Policy iteration (Howard's algorithm) for HJB equations:")
    print()
    print("  Given: H(x, p, α, m) = 0.5*α² + α·p + V(x,m)")
    print()
    print("  Algorithm:")
    print("    1. Start with initial policy α₀(x,t)")
    print("    2. Policy Evaluation: Solve HJB with α_k fixed")
    print("         ∂u/∂t + 0.5*α_k² + α_k·∂u/∂x + V(x,m) = 0")
    print("    3. Policy Improvement:")
    print("         α_{k+1} = argmax_α [α·∂u/∂x - 0.5*α²] = -∂u/∂x")
    print("    4. Check convergence: ||α_{k+1} - α_k|| < tol")
    print("    5. If not converged, go to step 2")
    print()
    print("  Advantages:")
    print("    - Superlinear convergence (faster than fixed-point)")
    print("    - Each iteration solves LINEAR PDE (not nonlinear)")
    print("    - Natural for control problems")
    print()
    print("=" * 80)
    print()


def solve_lq_mfg_with_value_iteration():
    """
    Solve a simple 1D LQ-MFG problem using standard value iteration (fixed-point).

    This serves as a baseline for comparison with policy iteration.
    """
    print("\n" + "=" * 80)
    print("Baseline: Value Iteration (Fixed-Point) Approach")
    print("=" * 80)

    # Create 1D LQ-MFG problem
    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[100 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=50,
        T=1.0,
        sigma=0.1,
        coupling_coefficient=0.5,
    )

    print("\nProblem setup:")
    bounds = problem.geometry.get_bounds()
    grid_shape = problem.geometry.get_grid_shape()
    print(f"  Domain: [{bounds[0][0]}, {bounds[1][0]}]")
    print(f"  Grid: {grid_shape[0]} spatial points")
    print(f"  Time: {problem.Nt + 1} time steps")
    print(f"  Diffusion: σ = {problem.sigma}")
    print(f"  Control cost: {problem.coupling_coefficient}")

    # Create HJB solver with fixed-point iteration
    solver = HJBFDMSolver(
        problem,
        solver_type="fixed_point",
        damping_factor=0.7,
        max_newton_iterations=100,
        newton_tolerance=1e-6,
    )

    # Create test density (uniform for simplicity)
    grid_shape = problem.geometry.get_grid_shape()
    M_test = np.ones((problem.Nt + 1, grid_shape[0])) / grid_shape[0]

    # Terminal condition: quadratic
    bounds = problem.geometry.get_bounds()
    x = np.linspace(bounds[0][0], bounds[1][0], grid_shape[0])
    U_terminal = 0.5 * (x - 0.5) ** 2  # Cost to deviate from center

    # Initial guess
    U_guess = np.zeros((problem.Nt + 1, grid_shape[0]))

    # Solve with value iteration
    print("\nSolving with value iteration (fixed-point)...")
    start_time = time.time()

    U_value_iteration = solver.solve_hjb_system(M_test, U_terminal, U_guess)

    value_iteration_time = time.time() - start_time

    print(f"  Time: {value_iteration_time:.3f}s")
    print(f"  Solution shape: {U_value_iteration.shape}")
    print(f"  Value at t=0, x=0.5: {U_value_iteration[0, 50]:.4f}")

    return U_value_iteration, value_iteration_time, problem, solver, M_test, U_terminal


def demonstrate_policy_iteration_structure():
    """
    Demonstrate the structure of policy iteration using the helper framework.

    Note: This shows how policy iteration WOULD be implemented. Full implementation
    requires solving the linearized HJB equation, which needs additional infrastructure.
    """
    print("\n" + "=" * 80)
    print("Policy Iteration Structure (Demonstration)")
    print("=" * 80)

    U_value, time_value, problem, solver, _M_test, _U_terminal = solve_lq_mfg_with_value_iteration()

    print("\nPolicy iteration workflow:")
    print("  1. Create policy problem wrapper")
    print("  2. Initialize policy (e.g., α₀ = 0)")
    print("  3. Iterate:")
    print("       - Evaluate policy (solve linear HJB)")
    print("       - Improve policy (α_{k+1} = -∇u_k)")
    print("       - Check convergence")
    print()

    # Create policy problem (conceptual)
    print("Creating policy iteration helper...")
    _policy_problem = create_lq_policy_problem(solver)

    print("  ✓ Policy problem created")
    print()
    print("Note: Full policy iteration requires solving the linearized HJB equation")
    print("      with fixed policy at each iteration. This requires additional")
    print("      infrastructure for handling the linear PDE solve.")
    print()

    # Demonstrate policy improvement step
    print("Demonstrating policy improvement step:")
    print("  Given: u(t,x) from value iteration")
    print("  Compute: α*(x,t) = -∂u/∂x")
    print()

    # Compute optimal policy from value function
    Nt, Nx_points = U_value.shape
    dx = problem.geometry.get_grid_spacing()[0]

    policy_from_value = np.zeros_like(U_value)

    for n in range(Nt):
        for i in range(Nx_points):
            if i == 0:
                dudx = (U_value[n, i + 1] - U_value[n, i]) / dx
            elif i == Nx_points - 1:
                dudx = (U_value[n, i] - U_value[n, i - 1]) / dx
            else:
                dudx = (U_value[n, i + 1] - U_value[n, i - 1]) / (2 * dx)

            policy_from_value[n, i] = -dudx

    print(f"  Policy computed: shape {policy_from_value.shape}")
    print(f"  Policy range: [{policy_from_value.min():.4f}, {policy_from_value.max():.4f}]")
    print()

    return U_value, policy_from_value, time_value


def visualize_results(U_value, policy, time_value):
    """Visualize value function and optimal policy."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping visualization")
        return

    print("\n" + "=" * 80)
    print("Visualization")
    print("=" * 80)

    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract dimensions
    Nt, Nx = U_value.shape
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 1, Nt)

    # Plot 1: Value function heatmap
    ax = axes[0, 0]
    im = ax.imshow(
        U_value,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="viridis",
    )
    ax.set_xlabel("Space (x)")
    ax.set_ylabel("Time (t)")
    ax.set_title("Value Function u(t,x)")
    plt.colorbar(im, ax=ax, label="u")

    # Plot 2: Policy heatmap
    ax = axes[0, 1]
    im = ax.imshow(
        policy,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="RdBu",
        vmin=-np.abs(policy).max(),
        vmax=np.abs(policy).max(),
    )
    ax.set_xlabel("Space (x)")
    ax.set_ylabel("Time (t)")
    ax.set_title("Optimal Policy α*(t,x)")
    plt.colorbar(im, ax=ax, label="α")

    # Plot 3: Value function at different times
    ax = axes[1, 0]
    time_indices = [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt - 1]
    for n in time_indices:
        ax.plot(x, U_value[n, :], label=f"t={t[n]:.2f}")
    ax.set_xlabel("Space (x)")
    ax.set_ylabel("Value u(t,x)")
    ax.set_title("Value Function Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Policy at different times
    ax = axes[1, 1]
    for n in time_indices:
        ax.plot(x, policy[n, :], label=f"t={t[n]:.2f}")
    ax.set_xlabel("Space (x)")
    ax.set_ylabel("Policy α(t,x)")
    ax.set_title("Optimal Policy Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "examples/outputs/policy_iteration_lq_demo.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def main():
    """Main demonstration."""
    print("\n" + "=" * 80)
    print(" POLICY ITERATION FOR LQ-MFG: DEMONSTRATION")
    print("=" * 80)
    print()
    print("This example demonstrates the concept and structure of policy iteration")
    print("(Howard's algorithm) for solving HJB equations in Mean Field Games.")
    print()
    print("Note: This is a conceptual demonstration showing how policy iteration")
    print("      would be applied. Full implementation requires solving the")
    print("      linearized HJB equation at each policy evaluation step.")
    print()

    # Step 1: Demonstrate concept
    demonstrate_policy_iteration_concept()

    # Step 2: Solve with value iteration (baseline)
    try:
        U_value, policy, time_value = demonstrate_policy_iteration_structure()

        # Step 3: Visualize results
        visualize_results(U_value, policy, time_value)

        # Summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print("\nThis demonstration showed:")
        print("  1. The concept of policy iteration for HJB equations")
        print("  2. How it differs from value iteration (fixed-point)")
        print("  3. The structure of policy evaluation and improvement")
        print("  4. Computing optimal policy from value function: α* = -∇u")
        print()
        print("Next steps for full implementation:")
        print("  - Implement linearized HJB solver with fixed policy")
        print("  - Add convergence monitoring for policy iteration")
        print("  - Compare convergence rates with value iteration")
        print("  - Extend to 2D/3D problems")
        print()
        print("=" * 80)
        print()

        return 0

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
