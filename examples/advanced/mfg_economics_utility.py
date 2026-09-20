#!/usr/bin/env python3
"""
Demonstration: expressing a utility MAXIMIZATION as the minimization this library solves.

The library is minimisation-only (#2373). There is no optimisation-direction switch, and
formulating the problem as a minimisation is the caller's job. This example is how to do that
job, using the case it comes up in most: an economist whose agents maximise utility.

The whole move is one line. Negate the utility.

    economist writes    U_util(x)          agents maximise this
    library wants       U_cost = -U_util   agents minimise this
    optimal control     alpha* = -grad(U_cost)/lambda

and the two descriptions are the same problem, not two conventions for it. That is exactly why
the direction parameter was removable: it never expressed anything a sign on the objective could
not, and carrying it meant every downstream consumer had to agree about which way it pointed.

This script runs BOTH columns to show they coincide:

  left   the economist's route -- write the utility, negate it, hand the cost to the library
  right  the control theorist's route -- write the cost directly

They produce the same optimal control and the same density evolution, to machine precision. If
they ever stop doing so, the negation identity above has broken and this example is the place it
shows up.

Setup: agents on [0, 1] prefer x = 0.7. The economist writes the utility as an inverted parabola
peaking there; the control theorist writes the cost as a parabola with its minimum there. Density
starts as a Gaussian at x = 0.2, away from the target, so the transport is visible.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid, no_flux_bc


def create_value_function(Nt: int, Nx: int, x: np.ndarray, perspective: str) -> np.ndarray:
    """Spatial preferences, written from either side.

    ``"utility"``  U_util(x) = -(x - 0.7)^2   an inverted parabola PEAKING at the target
    ``"cost"``     U_cost(x) =  (x - 0.7)^2   a parabola with its MINIMUM at the target

    These are the same preference. `main` negates the utility and hands the result to the
    library, which is the one step this example exists to show.
    """
    U = np.zeros((Nt + 1, Nx))
    shape = (x - 0.7) ** 2 if perspective == "cost" else -((x - 0.7) ** 2)
    U[:] = shape
    return U


def compute_gradient(U: np.ndarray, dx: float) -> np.ndarray:
    """Compute spatial gradient using central differences."""
    grad_U = np.zeros_like(U)
    Nt = U.shape[0]

    for t_idx in range(Nt):
        # Central differences in interior
        grad_U[t_idx, 1:-1] = (U[t_idx, 2:] - U[t_idx, :-2]) / (2 * dx)
        # One-sided at boundaries
        grad_U[t_idx, 0] = (U[t_idx, 1] - U[t_idx, 0]) / dx
        grad_U[t_idx, -1] = (U[t_idx, -1] - U[t_idx, -2]) / dx

    return grad_U


def solve_fp_with_drift(
    problem: MFGProblem,
    m0: np.ndarray,
    drift_field: np.ndarray,
    label: str,
) -> np.ndarray:
    """Solve FP equation with given drift field."""
    bc = no_flux_bc(dimension=1)
    fp_solver = FPFDMSolver(problem, boundary_conditions=bc)

    print(f"  Solving FP equation ({label})...")
    M = fp_solver.solve_fp_system(M_initial=m0, drift_field=drift_field, show_progress=False)

    return M


def analyze_results(
    x: np.ndarray,
    dx: float,
    M_minimize: np.ndarray,
    M_maximize: np.ndarray,
    target_location: float,
):
    """Analyze and compare results from both perspectives."""
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Mass conservation check
    for name, M in [("via utility", M_minimize), ("direct cost", M_maximize)]:
        mass_initial = np.trapezoid(M[0, :], dx=dx)
        mass_final = np.trapezoid(M[-1, :], dx=dx)
        print(f"\n{name} - Mass Conservation:")
        print(f"  Initial: {mass_initial:.6f}, Final: {mass_final:.6f}")
        print(f"  Deviation: {abs(mass_final - mass_initial):.2e}")

    # Center of mass evolution
    print("\nCenter of Mass Evolution:")
    for name, M in [("via utility", M_minimize), ("direct cost", M_maximize)]:
        com_initial = np.trapezoid(x * M[0, :], dx=dx) / np.trapezoid(M[0, :], dx=dx)
        com_final = np.trapezoid(x * M[-1, :], dx=dx) / np.trapezoid(M[-1, :], dx=dx)
        print(f"  {name}: {com_initial:.3f} -> {com_final:.3f} (target: {target_location})")

    # Final distribution comparison
    print("\nFinal Distribution Similarity:")
    diff = np.abs(M_minimize[-1, :] - M_maximize[-1, :])
    print(f"  Max difference: {diff.max():.6f}")
    print(f"  L2 difference: {np.sqrt(np.trapezoid(diff**2, dx=dx)):.6f}")


def plot_comparison(
    x: np.ndarray,
    m0: np.ndarray,
    U_minimize: np.ndarray,
    U_maximize: np.ndarray,
    grad_minimize: np.ndarray,
    grad_maximize: np.ndarray,
    alpha_minimize: np.ndarray,
    alpha_maximize: np.ndarray,
    M_minimize: np.ndarray,
    M_maximize: np.ndarray,
    target_location: float,
):
    """Create comprehensive comparison visualization."""
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)

    # Row 1: Value functions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, U_minimize[0, :], "b-", linewidth=2, label="U_util(x) = -(x-0.7)²")
    ax1.axvline(x=target_location, color="k", linestyle="--", alpha=0.5, label="Target x=0.7")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Value Function U")
    ax1.set_title("What the economist writes\nutility, PEAKS at the target")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, U_maximize[0, :], "r-", linewidth=2, label="U_cost(x) = (x-0.7)²")
    ax2.axvline(x=target_location, color="k", linestyle="--", alpha=0.5, label="Target x=0.7")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Value Function U")
    ax2.set_title("What the library wants\ncost, MINIMUM at the target")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Row 2: Gradients
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x, grad_minimize[0, :], "b-", linewidth=2, label="∇U")
    ax3.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax3.axvline(x=target_location, color="k", linestyle="--", alpha=0.5)
    ax3.set_xlabel("x")
    ax3.set_ylabel("Gradient ∇U")
    ax3.set_title("Gradient of the NEGATED utility")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x, grad_maximize[0, :], "r-", linewidth=2, label="∇U")
    ax4.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax4.axvline(x=target_location, color="k", linestyle="--", alpha=0.5)
    ax4.set_xlabel("x")
    ax4.set_ylabel("Gradient ∇U")
    ax4.set_title("Gradient of the direct cost")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Row 3: Optimal controls (THE KEY DIFFERENCE)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(x, alpha_minimize[0, :], "b-", linewidth=2.5, label="via negated utility", alpha=0.8)
    ax5.plot(x, alpha_maximize[0, :], "r--", linewidth=2.5, label="direct cost", alpha=0.8)
    ax5.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax5.axvline(x=target_location, color="k", linestyle="--", alpha=0.5, label="Target x=0.7")
    ax5.set_xlabel("x")
    ax5.set_ylabel("Optimal Control α*(x)")
    ax5.set_title(
        "Optimal Control Comparison\n"
        "Note: SAME physical direction (toward x=0.7) despite opposite formulas!\n"
        "This is because U_cost = -U_utility",
        fontsize=11,
    )
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)

    # Row 4: Density evolution heatmaps
    t_grid = np.linspace(0, 1, M_minimize.shape[0])
    extent = [x.min(), x.max(), t_grid.min(), t_grid.max()]

    ax6 = fig.add_subplot(gs[3, 0])
    im1 = ax6.imshow(M_minimize, aspect="auto", origin="lower", extent=extent, cmap="viridis", interpolation="bilinear")
    ax6.axvline(x=target_location, color="white", linestyle="--", alpha=0.7)
    ax6.set_xlabel("Space x")
    ax6.set_ylabel("Time t")
    ax6.set_title("Density m(t,x): via negated utility")
    plt.colorbar(im1, ax=ax6, label="Density")

    ax7 = fig.add_subplot(gs[3, 1])
    im2 = ax7.imshow(M_maximize, aspect="auto", origin="lower", extent=extent, cmap="viridis", interpolation="bilinear")
    ax7.axvline(x=target_location, color="white", linestyle="--", alpha=0.7)
    ax7.set_xlabel("Space x")
    ax7.set_ylabel("Time t")
    ax7.set_title("Density m(t,x): direct cost")
    plt.colorbar(im2, ax=ax7, label="Density")

    # Row 5: Final density comparison
    ax8 = fig.add_subplot(gs[4, :])
    ax8.plot(x, m0, "k--", linewidth=1.5, label="Initial m₀(x)", alpha=0.5)
    ax8.plot(x, M_minimize[-1, :], "b-", linewidth=2, label="via utility, final", alpha=0.8)
    ax8.plot(x, M_maximize[-1, :], "r--", linewidth=2, label="direct cost, final", alpha=0.8)
    ax8.axvline(x=target_location, color="k", linestyle="--", alpha=0.5, label="Target x=0.7")
    ax8.set_xlabel("x")
    ax8.set_ylabel("Density m(x)")
    ax8.set_title("Final Density Comparison (t = T)\nBoth perspectives yield identical physical evolution!")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.suptitle(
        "Expressing a utility maximisation as a minimisation: negate, then the routes coincide",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    return fig


def main():
    """Run economics utility maximization demonstration."""
    print("=" * 70)
    print("ECONOMICS UTILITY MAXIMIZATION DEMONSTRATION (Issue #623 Phase 3)")
    print("=" * 70)
    print("\nExpressing a utility MAXIMIZATION as the minimization the library solves.")
    print("Left: negate the utility. Right: write the cost directly. They must coincide.\n")

    # Setup
    Nx = 101
    Nt = 50
    T = 1.0
    sigma = 0.1
    control_cost_lambda = 1.0
    target_location = 0.7

    # The grid requires an explicit BC (no silent default). This line was missing and the
    # example had been dead on main before #2373 touched it -- unrelated to the removal.
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=no_flux_bc(dimension=1))
    # This example never solves an MFG -- it writes U by hand and runs only the FP half -- so the
    # problem is a carrier for the solver. The components are the minimum it now requires; that
    # requirement post-dates the example, which is a second reason it had stopped running on main.
    problem = MFGProblem(
        geometry=geometry,
        T=T,
        Nt=Nt,
        sigma=sigma,
        coupling_coefficient=0.0,  # No density coupling for clarity
        components=MFGComponents(
            m_initial=lambda x: float(np.exp(-100 * (x - 0.2) ** 2)),
            u_terminal=lambda x: 0.0,
            hamiltonian=SeparableHamiltonian(control_cost=QuadraticControlCost(control_cost=control_cost_lambda)),
        ),
    )

    x = np.linspace(0, 1, Nx)
    dx = x[1] - x[0]

    # Initial density: Gaussian at x=0.2 (away from target)
    print("Setting up initial density (Gaussian at x=0.2)...")
    m0 = np.exp(-100 * (x - 0.2) ** 2)
    m0 = m0 / np.trapezoid(m0, dx=dx)

    # ONE control cost. There is no direction to choose (#2373).
    print("\nCreating the control cost (one, minimisation-only):")
    cost = QuadraticControlCost(control_cost=control_cost_lambda)
    print("  alpha* = -grad(U_cost)/lambda")

    print("\nWriting the preference from both sides...")
    U_utility = create_value_function(Nt, Nx, x, "utility")
    U_cost_direct = create_value_function(Nt, Nx, x, "cost")
    print(f"  economist:        U_util(x) = -(x-{target_location})^2   (maximise this)")
    print(f"  control theorist: U_cost(x) =  (x-{target_location})^2   (minimise this)")

    # THE STEP THIS EXAMPLE IS ABOUT. The economist negates their utility; what comes out is a
    # cost, and from here the two routes are indistinguishable to the library.
    U_cost_from_utility = -U_utility
    print(
        "\n  negating the utility gives a cost identical to the direct one:",
        np.array_equal(U_cost_from_utility, U_cost_direct),
    )

    grad_from_utility = compute_gradient(U_cost_from_utility, dx)
    grad_direct = compute_gradient(U_cost_direct, dx)

    print("\nComputing optimal controls via ControlCostBase.optimal_control():")
    alpha_from_utility = cost.optimal_control(grad_from_utility)
    alpha_direct = cost.optimal_control(grad_direct)

    print(f"  via utility  alpha* at x=0.3: {alpha_from_utility[0, 30]:+.4f}  (positive, toward target)")
    print(f"  direct cost  alpha* at x=0.3: {alpha_direct[0, 30]:+.4f}  (positive, toward target)")
    agreement = float(np.max(np.abs(alpha_from_utility - alpha_direct)))
    print(f"  max|difference| between the two routes: {agreement:.3e}")
    assert agreement == 0.0, "the negation identity is broken -- see this file's docstring"

    print("\nSolving Fokker-Planck equations:")
    M_minimize = solve_fp_with_drift(problem, m0, alpha_from_utility, "via utility")
    M_maximize = solve_fp_with_drift(problem, m0, alpha_direct, "direct cost")

    # Analysis
    analyze_results(x, dx, M_minimize, M_maximize, target_location)

    # Visualization
    print("\nGenerating comparison plot...")
    fig = plot_comparison(
        x,
        m0,
        U_utility,
        U_cost_direct,
        grad_from_utility,
        grad_direct,
        alpha_from_utility,
        alpha_direct,
        M_minimize,
        M_maximize,
        target_location,
    )

    output_file = "mfg_economics_utility_comparison.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.show()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. The library minimises. There is no direction to select (#2373), and
   alpha* = -grad(U_cost)/lambda always.

2. To bring a maximisation problem to it, negate the objective ONCE, at
   the boundary of your own code: U_cost = -U_utility. Everything after
   that point is a minimisation and needs no further care.

3. This is not a convention the library imposes on your problem -- it is
   the same problem written the other way up. The run above asserts the
   two routes agree to 0.0, not merely to a tolerance.

4. Negate at the boundary, not per call site. The direction parameter was
   removed because it had to be agreed on by every downstream consumer,
   and a single negation on the objective expresses the same thing with
   nothing left to disagree about.

5. Problems that arrive as minimisations need no step at all:
   - Crowd evacuation (minimise travel cost)
   - Traffic flow (minimise congestion cost)
   - Robot swarms (minimise energy expenditure)

   Problems that arrive as maximisations take the one negation:
   - Economic models (maximise profit/utility)
   - Resource seeking (maximise resource collection)
   - Opinion dynamics (maximise social utility)

6. The control costs in mfgarchon/core/hamiltonian.py:
   - QuadraticControlCost: Standard L = ½λ|α|²
   - L1ControlCost: Bang-bang control L = λ|α|
   - BoundedControlCost: Constrained control |α| ≤ α_max
    """)


if __name__ == "__main__":
    main()
