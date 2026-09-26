#!/usr/bin/env python3
"""
Demonstration: L1 Control Cost in Mean Field Games (Issue #573)

This example demonstrates how to use the FP solver with non-quadratic Hamiltonians,
specifically L1 control cost (minimal fuel).

Mathematical Setup:
-------------------
Standard MFG (quadratic control):
    H(p) = (1/2)|p|²  →  α*(x) = -∇U(x)

L1 control (minimal fuel):
    H(p) = |p|  →  α*(x) = -sign(∇U(x))

Key Difference:
- Quadratic: Control magnitude proportional to gradient
- L1: Control is bang-bang (constant magnitude, switches direction)

This example compares both control laws applied to the same value function.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.geometry import TensorProductGrid, no_flux_bc


def create_synthetic_value_function(problem: MFGProblem) -> np.ndarray:
    """
    Create a synthetic value function U(t,x) for demonstration.

    We use U(x) = (x - 0.5)² which creates a potential well at x=0.5.
    This represents the "cost to go" from position x.
    """
    Nt, Nx = problem.Nt + 1, problem.geometry.get_grid_shape()[0]
    x = np.linspace(0, 1, Nx)

    U = np.zeros((Nt, Nx))
    for t_idx in range(Nt):
        # Quadratic potential centered at x=0.5
        U[t_idx, :] = (x - 0.5) ** 2
        # Could add time-varying component:
        # U[t_idx, :] *= (1 - t_idx / Nt)

    return U


def compute_gradient_1d(U: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute spatial gradient of U using central differences.

    Args:
        U: Value function, shape (Nt+1, Nx)
        dx: Spatial grid spacing

    Returns:
        Gradient field, shape (Nt+1, Nx)
    """
    Nt, _ = U.shape
    grad_U = np.zeros_like(U)

    for t_idx in range(Nt):
        # Central differences in interior
        grad_U[t_idx, 1:-1] = (U[t_idx, 2:] - U[t_idx, :-2]) / (2 * dx)
        # Forward/backward at boundaries
        grad_U[t_idx, 0] = (U[t_idx, 1] - U[t_idx, 0]) / dx
        grad_U[t_idx, -1] = (U[t_idx, -1] - U[t_idx, -2]) / dx

    return grad_U


def compute_quadratic_control(grad_U: np.ndarray) -> np.ndarray:
    """
    Quadratic Hamiltonian: H = (1/2)|p|²

    Optimal control: α* = -∇U
    """
    return -grad_U


def compute_l1_control(grad_U: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    L1 Hamiltonian: H = |p|

    Optimal control: α* = -sign(∇U)

    Args:
        grad_U: Gradient field
        epsilon: Regularization to avoid division by zero

    Returns:
        L1 optimal control (bang-bang)
    """
    # Pure L1: α* = -sign(∇U)
    # Note: At ∇U = 0, the optimal control is not unique (any α with |α| ≤ 1 works)
    # We use sign function which returns 0 at zero
    return -np.sign(grad_U)


def compute_quartic_control(grad_U: np.ndarray) -> np.ndarray:
    """
    Quartic Hamiltonian: H = (1/4)|p|⁴

    Optimal control: α* = -|∇U|^(1/3) * sign(∇U) = -(∇U)^(1/3)
    """
    # Preserve sign while taking cube root of magnitude
    return -np.sign(grad_U) * np.abs(grad_U) ** (1.0 / 3.0)


def solve_fp_with_control(
    problem: MFGProblem,
    m0: np.ndarray,
    drift_field: np.ndarray,
    control_name: str,
) -> np.ndarray:
    """Solve FP equation with given drift field."""
    bc = no_flux_bc(dimension=1)
    fp_solver = FPFDMSolver(problem, boundary_conditions=bc)

    print(f"\nSolving FP equation with {control_name} control...")
    M = fp_solver.solve_fp_system(M_initial=m0, drift_field=drift_field, show_progress=False)

    return M


def analyze_mass_conservation(M: np.ndarray, dx: float, control_name: str):
    """Check mass conservation."""
    Nt = M.shape[0]
    masses = np.array([np.trapezoid(M[t, :], dx=dx) for t in range(Nt)])

    mass_initial = masses[0]
    mass_final = masses[-1]
    max_deviation = np.max(np.abs(masses - mass_initial))

    print(f"\n{control_name} Control - Mass Conservation:")
    print(f"  Initial mass: {mass_initial:.6f}")
    print(f"  Final mass:   {mass_final:.6f}")
    print(f"  Max deviation: {max_deviation:.2e}")


def plot_comparison(
    x: np.ndarray,
    U: np.ndarray,
    grad_U: np.ndarray,
    alpha_quadratic: np.ndarray,
    alpha_L1: np.ndarray,
    alpha_quartic: np.ndarray,
    M_quadratic: np.ndarray,
    M_L1: np.ndarray,
    M_quartic: np.ndarray,
    m0: np.ndarray,
):
    """Create comprehensive comparison visualization."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # Row 1: Value function and gradient
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, U[0, :], "b-", linewidth=2, label="U(x)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Value Function U")
    ax1.set_title("Value Function U(x) = (x - 0.5)²")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, grad_U[0, :], "g-", linewidth=2, label="∇U(x)")
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Gradient ∇U")
    ax2.set_title("Gradient Field ∇U(x)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(x, m0, "purple", linewidth=2, label="m₀(x)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("Initial Density")
    ax3.set_title("Initial Condition m₀(x)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Row 2: Optimal controls comparison
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(x, alpha_quadratic[0, :], "b-", linewidth=2, label="Quadratic: α* = -∇U", alpha=0.7)
    ax4.plot(x, alpha_L1[0, :], "r-", linewidth=2, label="L1: α* = -sign(∇U)", alpha=0.7)
    ax4.plot(x, alpha_quartic[0, :], "orange", linewidth=2, label="Quartic: α* = -(∇U)^(1/3)", alpha=0.7)
    ax4.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax4.set_xlabel("x")
    ax4.set_ylabel("Optimal Control α*(x)")
    ax4.set_title("Optimal Control Comparison: Different Hamiltonians")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    # Row 3: Density evolution (space-time heatmaps)
    t_grid = np.linspace(0, 1, M_quadratic.shape[0])
    extent = [x.min(), x.max(), t_grid.min(), t_grid.max()]

    ax5 = fig.add_subplot(gs[2, 0])
    im1 = ax5.imshow(
        M_quadratic, aspect="auto", origin="lower", extent=extent, cmap="viridis", interpolation="bilinear"
    )
    ax5.set_xlabel("Space x")
    ax5.set_ylabel("Time t")
    ax5.set_title("Quadratic Control\nm(t,x) Evolution")
    plt.colorbar(im1, ax=ax5, label="Density")

    ax6 = fig.add_subplot(gs[2, 1])
    im2 = ax6.imshow(M_L1, aspect="auto", origin="lower", extent=extent, cmap="viridis", interpolation="bilinear")
    ax6.set_xlabel("Space x")
    ax6.set_ylabel("Time t")
    ax6.set_title("L1 Control\nm(t,x) Evolution")
    plt.colorbar(im2, ax=ax6, label="Density")

    ax7 = fig.add_subplot(gs[2, 2])
    im3 = ax7.imshow(M_quartic, aspect="auto", origin="lower", extent=extent, cmap="viridis", interpolation="bilinear")
    ax7.set_xlabel("Space x")
    ax7.set_ylabel("Time t")
    ax7.set_title("Quartic Control\nm(t,x) Evolution")
    plt.colorbar(im3, ax=ax7, label="Density")

    # Row 4: Final density comparison
    ax8 = fig.add_subplot(gs[3, :])
    ax8.plot(x, m0, "k--", linewidth=1.5, label="Initial m₀(x)", alpha=0.5)
    ax8.plot(x, M_quadratic[-1, :], "b-", linewidth=2, label="Quadratic (t=T)", alpha=0.7)
    ax8.plot(x, M_L1[-1, :], "r-", linewidth=2, label="L1 (t=T)", alpha=0.7)
    ax8.plot(x, M_quartic[-1, :], "orange", linewidth=2, label="Quartic (t=T)", alpha=0.7)
    ax8.set_xlabel("x")
    ax8.set_ylabel("Density m(x)")
    ax8.set_title("Final Density Distribution Comparison (t = T)")
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=10)

    plt.suptitle(
        "L1 Control vs Quadratic Control vs Quartic Control in Mean Field Games",
        fontsize=14,
        fontweight="bold",
    )

    return fig


def main():
    """Run L1 control demonstration."""
    print("=" * 80)
    print("L1 CONTROL COST DEMONSTRATION (Issue #573)")
    print("=" * 80)

    # Setup problem
    print("\nSetting up 1D MFG problem...")
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[101])
    problem = MFGProblem(
        geometry=geometry,
        T=1.0,
        Nt=50,
        volatility=0.1,  # Diffusion coefficient
        coupling_coefficient=0.0,  # No coupling for this demo
    )

    x = np.linspace(0, 1, 101)
    dx = x[1] - x[0]

    # Initial density: Gaussian at x=0.2
    print("Creating initial density (Gaussian at x=0.2)...")
    m0 = np.exp(-100 * (x - 0.2) ** 2)
    m0 = m0 / np.trapezoid(m0, dx=dx)  # Normalize

    # Create synthetic value function
    print("Creating synthetic value function U(x) = (x - 0.5)²...")
    U = create_synthetic_value_function(problem)

    # Compute gradient
    print("Computing gradient ∇U...")
    grad_U = compute_gradient_1d(U, dx)

    # Compute optimal controls for different Hamiltonians
    print("\nComputing optimal controls:")
    print("  - Quadratic: α* = -∇U")
    print("  - L1:        α* = -sign(∇U)")
    print("  - Quartic:   α* = -(∇U)^(1/3)")

    alpha_quadratic = compute_quadratic_control(grad_U)
    alpha_L1 = compute_l1_control(grad_U)
    alpha_quartic = compute_quartic_control(grad_U)

    # Solve FP equations
    M_quadratic = solve_fp_with_control(problem, m0, alpha_quadratic, "Quadratic")
    M_L1 = solve_fp_with_control(problem, m0, alpha_L1, "L1")
    M_quartic = solve_fp_with_control(problem, m0, alpha_quartic, "Quartic")

    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    analyze_mass_conservation(M_quadratic, dx, "Quadratic")
    analyze_mass_conservation(M_L1, dx, "L1")
    analyze_mass_conservation(M_quartic, dx, "Quartic")

    # Control characteristics
    print("\nControl Characteristics:")
    print(f"  Quadratic control - Range: [{alpha_quadratic.min():.3f}, {alpha_quadratic.max():.3f}]")
    print(f"  L1 control        - Range: [{alpha_L1.min():.3f}, {alpha_L1.max():.3f}] (bang-bang)")
    print(f"  Quartic control   - Range: [{alpha_quartic.min():.3f}, {alpha_quartic.max():.3f}] (weaker)")

    # Density evolution
    print("\nDensity Evolution:")
    for name, M in [("Quadratic", M_quadratic), ("L1", M_L1), ("Quartic", M_quartic)]:
        peak_initial = m0.max()
        peak_final = M[-1, :].max()
        spread = np.sqrt(np.trapezoid((x - 0.5) ** 2 * M[-1, :], dx=dx))
        print(f"  {name:10s} - Peak: {peak_initial:.3f} → {peak_final:.3f}, Spread: {spread:.3f}")

    # Visualization
    print("\nGenerating comparison plot...")
    fig = plot_comparison(x, U, grad_U, alpha_quadratic, alpha_L1, alpha_quartic, M_quadratic, M_L1, M_quartic, m0)

    output_file = "mfg_l1_control_comparison.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.show()

    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS")
    print("=" * 80)
    print("""
1. Control Laws:
   - Quadratic: Smooth, proportional to gradient
   - L1:        Bang-bang (constant ±1), switches at ∇U = 0
   - Quartic:   Weaker response than quadratic (sublinear)

2. Physical Interpretation:
   - Quadratic: Standard control with quadratic cost
   - L1:        Minimal fuel (constant thrust, only direction changes)
   - Quartic:   More expensive control (discourages large actions)

3. Density Dynamics:
   - All controls drive density toward x=0.5 (minimum of U)
   - L1 creates sharper transitions (bang-bang nature)
   - Quartic creates slower evolution (weaker control)

4. Implementation (Issue #573):
   The drift_field parameter in FP solvers accepts drift velocity
   for ANY Hamiltonian - just compute α* = -∂_p H(∇U) and pass it in!
    """)


if __name__ == "__main__":
    main()
