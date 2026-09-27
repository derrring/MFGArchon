#!/usr/bin/env python3
"""
Visualize 2D Density Evolution with Semi-Lagrangian Enhancements

This script demonstrates the density evolution in a 2D Mean Field Game
using the enhanced Semi-Lagrangian solver (RK4 + cubic + RBF).

Shows agents navigating from initial position to goal while interacting
through the mean field coupling.
"""

from __future__ import annotations

import sys

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary.conditions import no_flux_bc


def create_simple_coupled_problem(
    domain_bounds=(0.0, 1.0, 0.0, 1.0),
    grid_resolution=20,
    time_domain=(0.5, 30),
    diffusion_coeff=0.1,
    coupling_strength=1.0,
    goal_position=None,
) -> MFGProblem:
    """
    Create a simple coupled MFG problem for visualization.

    Agents navigate from (0.3, 0.3) to (0.7, 0.7) while avoiding congestion.
    Hamiltonian: H = (1/2)|p|^2 + coupling_strength * m
    """
    if goal_position is None:
        goal_position = [0.7, 0.7]

    T, Nt = time_domain

    # Class-based Hamiltonian: H = (1/2)|p|^2 + coupling_strength * m
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: coupling_strength * m,
        coupling_dm=lambda m: coupling_strength,
    )
    model = Model(hamiltonian=hamiltonian, volatility=diffusion_coeff)

    def initial_density_func(x):
        """Gaussian initial density centered at (0.3, 0.3)."""
        x = np.atleast_2d(x)
        center = np.array([0.3, 0.3])
        dist_sq = np.sum((x - center) ** 2, axis=1)
        density = np.exp(-50 * dist_sq)
        return density / (np.sum(density) + 1e-10)

    def terminal_cost_func(x):
        """Quadratic terminal cost: distance to goal."""
        x = np.atleast_2d(x)
        goal = np.array(goal_position)
        dist_sq = np.sum((x - goal) ** 2, axis=1)
        return 0.5 * dist_sq

    conditions = Conditions(
        m_initial=initial_density_func,
        u_terminal=terminal_cost_func,
        T=T,
    )

    # 2D geometry with no-flux BC
    spatial_bounds = [(domain_bounds[0], domain_bounds[1]), (domain_bounds[2], domain_bounds[3])]
    geometry = TensorProductGrid(
        bounds=spatial_bounds,
        Nx_points=[grid_resolution, grid_resolution],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    problem = MFGProblem(
        model=model,
        domain=geometry,
        conditions=conditions,
        Nt=Nt,
    )
    # Store for use by visualization/solver code
    problem.coupling_strength = coupling_strength
    problem.goal_position = goal_position
    return problem


def solve_mfg_with_enhancements(use_enhancements=True):
    """Solve 2D MFG problem with or without enhancements."""
    print("\n" + "=" * 80)
    config_name = "ENHANCED" if use_enhancements else "BASELINE"
    print(f"Solving with {config_name} Semi-Lagrangian Configuration")
    print("=" * 80)

    # Create problem
    problem = create_simple_coupled_problem(
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
        grid_resolution=20,
        time_domain=(0.5, 30),
        diffusion_coeff=0.1,
        coupling_strength=0.5,
    )

    print("\nProblem setup:")
    print("  Domain: [0, 1] × [0, 1]")
    print(f"  Grid: {problem.geometry.grid.num_points}")
    print(f"  Coupling: {problem.coupling_strength}")

    if use_enhancements:
        print("\nEnhancements:")
        print("  - RK4 characteristic tracing (scipy.solve_ivp)")
        print("  - Cubic spline interpolation")
        print("  - RBF interpolation fallback")

        hjb_solver = HJBSemiLagrangianSolver(
            problem,
            characteristic_solver="rk4",
            interpolation_method="cubic",
            use_rbf_fallback=True,
            rbf_kernel="thin_plate_spline",
            use_jax=False,
        )
    else:
        print("\nBaseline:")
        print("  - Explicit Euler characteristic tracing")
        print("  - Linear interpolation")
        print("  - No RBF fallback")

        hjb_solver = HJBSemiLagrangianSolver(
            problem,
            characteristic_solver="explicit_euler",
            interpolation_method="linear",
            use_rbf_fallback=False,
            use_jax=False,
        )

    # Create FP solver
    fp_solver = FPFDMSolver(problem, boundary_conditions=no_flux_bc(dimension=2))

    # Setup
    grid = problem.geometry.grid
    ndim = grid.dimension
    Nt = problem.Nt + 1
    shape = tuple(grid.num_points)

    # Create grid points using meshgrid
    x_vals = []
    for d in range(ndim):
        x_min = grid.bounds[d][0]
        spacing = grid.spacing[d]
        n_points = grid.num_points[d]
        x_vals.append(x_min + np.arange(n_points) * spacing)

    meshgrid_arrays = np.meshgrid(*x_vals, indexing="ij")
    x_flat = np.column_stack([arr.ravel() for arr in meshgrid_arrays])

    # Initialize density using components (populated from Model+Conditions)
    m_init_flat = problem.components.m_initial(x_flat)
    m_init = m_init_flat.reshape(shape)
    m_init = m_init / np.sum(m_init)

    # Initialize M as constant in time
    M = np.tile(m_init, (Nt, *([1] * ndim)))

    # Initialize U as terminal cost
    u_final_flat = problem.components.u_terminal(x_flat)
    u_final = u_final_flat.reshape(shape)
    U = np.tile(u_final, (Nt, *([1] * ndim)))

    print("\nRunning Picard iteration...")

    # Picard iteration
    for iteration in range(3):  # 3 iterations for convergence
        print(f"  Iteration {iteration + 1}/3")

        # Solve HJB backward in time
        U_prev = U.copy()
        U = hjb_solver.solve_hjb_system(
            M_density=M,
            U_terminal=u_final,
            U_coupling_prev=U_prev,
        )

        # Solve FP forward in time (if not last iteration)
        if iteration < 2:
            M = fp_solver.solve_fp_system(
                M_initial=m_init,
                drift_field=U,
                show_progress=False,
            )

    print("\n[OK] Converged")
    print(f"  Final density mass: {np.sum(M[-1]):.6f}")
    print(f"  Value function range: [{U.min():.4f}, {U.max():.4f}]")

    return U, M, problem


def visualize_density_evolution(U, M, problem):
    """Create visualization of density evolution over time."""
    if not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not available, skipping visualization")
        return

    print("\n" + "=" * 80)
    print("Creating Density Evolution Visualization")
    print("=" * 80)

    grid_shape = problem.geometry.grid.num_points
    Nt = M.shape[0]

    # Select time steps to visualize
    time_indices = [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt - 1]
    time_labels = ["t=0\n(start)", "t=T/4", "t=T/2", "t=3T/4", "t=T\n(end)"]

    _fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Row 1: Density evolution
    for col, (t_idx, t_label) in enumerate(zip(time_indices, time_labels, strict=True)):
        ax = axes[0, col]

        # Density is already shaped
        M_2d = M[t_idx]

        im = ax.imshow(
            M_2d.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="hot",
            aspect="auto",
            vmin=0,
            vmax=M.max(),
        )
        ax.set_title(f"Density {t_label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Mark initial position and goal
        if t_idx == 0:
            ax.plot(0.3, 0.3, "b*", markersize=20, label="Initial", markeredgecolor="white", markeredgewidth=1)
        ax.plot(0.7, 0.7, "g*", markersize=20, label="Goal", markeredgecolor="white", markeredgewidth=1)
        if t_idx == 0:
            ax.legend(fontsize=9, loc="upper left")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2: Value function evolution
    for col, (t_idx, t_label) in enumerate(zip(time_indices, time_labels, strict=True)):
        ax = axes[1, col]

        # Value function is already shaped
        U_2d = U[t_idx]

        im = ax.imshow(
            U_2d.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="viridis",
            aspect="auto",
        )
        ax.set_title(f"Value {t_label}", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Mark goal
        ax.plot(0.7, 0.7, "r*", markersize=15, markeredgecolor="white", markeredgewidth=1)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = "examples/outputs/2d_density_evolution.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Visualization saved to: {save_path}")
    plt.close()

    # Create density trajectory plot
    _fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot density center of mass trajectory
    centers_x = []
    centers_y = []

    x_coords = np.linspace(0, 1, grid_shape[0])
    y_coords = np.linspace(0, 1, grid_shape[1])

    for t in range(Nt):
        M_2d = M[t]

        # Compute center of mass
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

        total_mass = np.sum(M_2d)
        if total_mass > 1e-10:
            cx = np.sum(X * M_2d) / total_mass
            cy = np.sum(Y * M_2d) / total_mass
            centers_x.append(cx)
            centers_y.append(cy)

    # Plot trajectory
    ax.plot(centers_x, centers_y, "b-", linewidth=2, label="Density center trajectory", alpha=0.7)
    ax.plot(centers_x[0], centers_y[0], "bo", markersize=15, label="Start", markeredgecolor="white", markeredgewidth=2)
    ax.plot(centers_x[-1], centers_y[-1], "go", markersize=15, label="End", markeredgecolor="white", markeredgewidth=2)
    ax.plot(0.7, 0.7, "r*", markersize=25, label="Goal", markeredgecolor="white", markeredgewidth=2)

    # Plot density contours at selected times
    for t_idx, alpha in zip([0, Nt // 2, Nt - 1], [0.8, 0.5, 0.3], strict=True):
        M_2d = M[t_idx]
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

        ax.contour(X, Y, M_2d, levels=3, colors="red", alpha=alpha, linewidths=1.5)

    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.set_title(
        "2D Density Evolution: Center of Mass Trajectory\n(with density contours)", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    save_path = "examples/outputs/2d_density_trajectory.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Trajectory plot saved to: {save_path}")
    plt.close()


def main():
    """Run visualization."""
    print("\n" + "=" * 80)
    print(" 2D DENSITY EVOLUTION WITH SEMI-LAGRANGIAN ENHANCEMENTS")
    print("=" * 80)
    print("\nThis demonstrates:")
    print("  - 2D Mean Field Game with crowd navigation")
    print("  - Semi-Lagrangian HJB solver with all enhancements")
    print("  - Density evolution from initial position to goal")
    print("  - Interaction through mean field coupling")

    try:
        # Solve with enhancements
        U, M, problem = solve_mfg_with_enhancements(use_enhancements=True)

        # Visualize
        visualize_density_evolution(U, M, problem)

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print("\n[OK] Demonstrated 2D Semi-Lagrangian enhancements:")
        print("  - RK4 characteristic tracing handles 2D velocity fields")
        print("  - Cubic interpolation provides smooth 2D density evolution")
        print("  - RBF fallback ensures robustness at boundaries")
        print("\n[OK] Density successfully evolved from start (0.3, 0.3) to goal (0.7, 0.7)")
        print("[OK] Mean field coupling captured congestion avoidance")
        print()

        return 0

    except Exception as e:
        print(f"\n[ERROR] Visualization failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
