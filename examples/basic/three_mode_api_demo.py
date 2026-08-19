"""
Three-Mode Solving API Demo (Issue #580)

This example demonstrates the three ways to solve MFG problems with guaranteed
adjoint duality between HJB and FP solvers.

References:
    - Issue #580: Adjoint-aware solver pairing
"""

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types import NumericalScheme


def create_lq_model_and_conditions():
    """Create standard LQ-MFG model and conditions for demos."""
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.5 * m,
        coupling_dm=lambda m: 0.5,
    )
    model = Model(hamiltonian=hamiltonian, sigma=0.1)
    conditions = Conditions(
        u_terminal=lambda x: (x - 0.5) ** 2,
        m_initial=lambda x: np.exp(-50 * (x - 0.5) ** 2),
        T=1.0,
    )
    return model, conditions


def _fmt_mass_error(result) -> str:
    """`mass_conservation_error` is None when the geometry has no volume element, or when the
    coupling loop broke on iteration 1 with a non-finite U -- in which case `solve` returns
    normally and formatting it would raise TypeError on the very line meant to make failure
    visible."""
    err = result.mass_conservation_error
    return "not measured" if err is None else f"{err:.3e}"


def create_problem():
    """Create a simple 1D problem for demos."""
    domain = TensorProductGrid(
        bounds=[(0.0, 1.0)],
        Nx_points=[40],
        boundary_conditions=no_flux_bc(dimension=1),
    )
    model, conditions = create_lq_model_and_conditions()
    return MFGProblem(
        model=model,
        domain=domain,
        conditions=conditions,
        Nt=20,
    )


def demo_safe_mode():
    """
    Safe Mode: Specify numerical scheme for automatic dual pairing.

    Benefits:
    - Guaranteed adjoint duality by construction
    - No way to accidentally mix incompatible solvers
    - Recommended for most users
    """
    print("\n" + "=" * 70)
    print("SAFE MODE: Automatic Dual Pairing")
    print("=" * 70)

    # Create a simple 1D problem
    problem = create_problem()

    # Safe Mode: Just specify the scheme
    result = problem.solve(
        scheme=NumericalScheme.FDM_UPWIND,
        max_iterations=60,
        tolerance=1e-6,
        verbose=True,
    )

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final error (max): {result.max_error:.3e}")
    print(f"Mass conservation error: {_fmt_mass_error(result)}")

    return result


def demo_expert_mode():
    """
    Expert Mode: Manual solver injection with duality validation.

    Benefits:
    - Full control over solver configuration
    - Duality validation with educational warnings
    - Useful for advanced customization
    """
    print("\n" + "=" * 70)
    print("EXPERT MODE: Manual Solver Injection")
    print("=" * 70)

    problem = create_problem()

    # Expert Mode: Create and configure solvers manually.
    # 'divergence_upwind' is the only one of the four advection schemes that solves this problem.
    # Measured here, cap 60, one outcome each -- the failures are NOT the same failure:
    #   gradient_centered    aborts at timestep 5/20 (density -6.388e-06)
    #   gradient_upwind      converges at 27, and loses 98.17% of the mass
    #   divergence_centered  aborts at timestep 3/20 (density -6.319e-06)
    #   divergence_upwind    converges at 49, mass conserved to 5.1e-15
    # The gradient family is non-conservative at a no-flux wall (#1075, #2007, #2008), but HOW a
    # given one fails is a property of the configuration, not of the family -- see the leak table
    # in geometry/boundary/conditions.py, which says so and is measured at a different resolution.
    hjb = HJBFDMSolver(problem)
    fp = FPFDMSolver(problem, advection_scheme="divergence_upwind")

    # Solve with custom solvers (duality automatically validated).
    # 49 iterations, against 27 for the leaking scheme: losing the mass made the problem easier.
    # What hid the loss is that err_M measures an INCREMENT. At this cap the leaking run reports
    # converged=True at iteration 27 with err_M = 6.43e-09 and 98% of the mass gone, simultaneously.
    # The loop did measure the mass (result.mass_conservation_error = 0.98), but nothing gates on
    # it and nothing printed it, which is why the line below now does.
    result = problem.solve(
        hjb_solver=hjb,
        fp_solver=fp,
        max_iterations=60,
        tolerance=1e-6,
        verbose=True,
    )

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final error (max): {result.max_error:.3e}")
    print(f"Mass conservation error: {_fmt_mass_error(result)}")

    return result


def demo_auto_mode():
    """
    Auto Mode: Intelligent automatic scheme selection.

    Benefits:
    - Zero configuration required
    - Analyzes problem geometry and selects appropriate scheme
    - Good default for quick experiments
    """
    print("\n" + "=" * 70)
    print("AUTO MODE: Intelligent Automatic Selection")
    print("=" * 70)

    problem = create_problem()

    # Auto Mode: No scheme or solvers specified
    # System automatically selects FDM_UPWIND (safe default)
    result = problem.solve(
        max_iterations=60,
        tolerance=1e-6,
        verbose=True,
    )

    print(f"\nConverged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final error (max): {result.max_error:.3e}")
    print(f"Mass conservation error: {_fmt_mass_error(result)}")

    return result


def compare_schemes():
    """
    Compare different numerical schemes in Safe Mode.

    Shows how easy it is to test different discretizations while
    maintaining adjoint duality guarantees.
    """
    print("\n" + "=" * 70)
    print("SCHEME COMPARISON: Testing Different Discretizations")
    print("=" * 70)

    problem = create_problem()

    schemes = [
        NumericalScheme.FDM_UPWIND,
        NumericalScheme.FDM_CENTERED,
    ]

    results = {}

    for scheme in schemes:
        print(f"\n--- Testing {scheme.value} ---")
        try:
            result = problem.solve(
                scheme=scheme,
                max_iterations=60,
                tolerance=1e-6,
                verbose=False,
            )
        except ValueError as exc:
            # Narrow deliberately. The mass-fabrication gate raises a bare ValueError
            # (utils/numerical/mass_fabrication_gate.py), so its own wording is the only thing that
            # distinguishes a principled refusal -- the centered schemes are not
            # positivity-preserving, and the library declines to clip a negative density rather
            # than report a conserved one it did not compute. Everything else on this path also
            # raises ValueError (NaN/Inf blow-ups, unknown-scheme dispatch), and catching those
            # here would print a library bug under the word "refused" and still exit 0.
            if "fabricate" not in str(exc):
                raise
            results[scheme.value] = None
            print(f"Refused (mass-fabrication gate): {exc}")
            continue
        results[scheme.value] = result
        print(f"Converged: {result.converged} in {result.iterations} iterations")
        print(f"Final error (max): {result.max_error:.3e}")
        print(f"Mass conservation error: {_fmt_mass_error(result)}")

    return results


def plot_results(result, problem):
    """Visualize the solution from any mode."""
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)

    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    # SolverResult has no `.problem`; take the time horizon from the problem.
    t = np.linspace(0.0, problem.T, result.U.shape[0])

    # Plot value function
    ax = axes[0]
    x = np.linspace(0, 1, result.U.shape[1])
    for t_idx in [0, result.U.shape[0] // 2, -1]:
        ax.plot(x, result.U[t_idx, :], label=f"t={t[t_idx]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("U(t, x)")
    ax.set_title("Value Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot density
    ax = axes[1]
    for t_idx in [0, result.M.shape[0] // 2, -1]:
        ax.plot(x, result.M[t_idx, :], label=f"t={t[t_idx]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("m(t, x)")
    ax.set_title("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Plots displayed.")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "THREE-MODE SOLVING API DEMO" + " " * 25 + "║")
    print("║" + " " * 20 + "(Issue #580)" + " " * 35 + "║")
    print("╚" + "=" * 68 + "╝")

    print("\nThis demo shows three ways to solve MFG problems:")
    print("  1. Safe Mode: Specify scheme for automatic dual pairing")
    print("  2. Expert Mode: Manual solver injection with validation")
    print("  3. Auto Mode: Intelligent automatic selection")
    print("\nAll three modes guarantee adjoint duality between HJB and FP solvers.")

    # Run demos
    result_safe = demo_safe_mode()
    result_expert = demo_expert_mode()
    result_auto = demo_auto_mode()

    # Compare schemes
    compare_schemes()

    # Visualize one result
    plot_results(result_safe, create_problem())

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nAll three modes produced solutions:")
    print(f"  Safe Mode:   {result_safe.iterations} iterations, error={result_safe.max_error:.3e}")
    print(f"  Expert Mode: {result_expert.iterations} iterations, error={result_expert.max_error:.3e}")
    print(f"  Auto Mode:   {result_auto.iterations} iterations, error={result_auto.max_error:.3e}")

    print("\nThe three are identical here to every printed digit, and that is the point: they are")
    print("three routes to the SAME solver pair, not three algorithms. The mode decides how much")
    print("you specify, not what gets solved.")

    print("\nRecommendation: Use Safe Mode for most applications.")
    print("                Use Expert Mode only when you need custom solver config.")
    print("                Use Auto Mode for quick experiments with default settings.")


if __name__ == "__main__":
    main()
