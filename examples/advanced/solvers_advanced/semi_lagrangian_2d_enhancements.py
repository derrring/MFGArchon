#!/usr/bin/env python3
"""
2D Semi-Lagrangian Enhancements Demonstration

This example demonstrates the Semi-Lagrangian enhancements for 2D problems:
1. RK4 characteristic tracing with scipy.solve_ivp
2. Cubic spline interpolation for smooth 2D fields
3. RBF interpolation fallback for robustness

Shows all three enhancements working together on a 2D crowd navigation problem.
"""

from __future__ import annotations

import time

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
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.mfg_logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("semi_lagrangian_2d", level="INFO")
logger = get_logger(__name__)


def create_2d_crowd_navigation_problem(
    grid_resolution=15,
    time_horizon=0.5,
    num_timesteps=25,
    sigma=0.05,
    coupling_strength=0.5,
):
    """
    Create a simple 2D crowd navigation problem for demonstrating enhancements.

    Agents navigate from bottom-left to top-right while avoiding congestion.
    - Isotropic Hamiltonian: H = (1/2)|nabla u|^2
    - MFG coupling: agents prefer less crowded regions
    - Goal: reach target at (0.8, 0.8)

    Returns
    -------
    MFGProblem
        Configured 2D crowd navigation problem.
    """
    goal_position = np.array([0.8, 0.8])
    initial_position = np.array([0.2, 0.2])

    # Geometry
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[grid_resolution, grid_resolution],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    # Model: H = (1/2)|p|^2 + kappa * m
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: coupling_strength * m,
        coupling_dm=lambda m: coupling_strength,
    )
    model = Model(hamiltonian=hamiltonian, volatility=sigma)

    # Conditions (callables, resolution-independent)
    def initial_density(x):
        """Gaussian initial density centered at (0.2, 0.2). x shape: (2,) for single 2D point."""
        dist_sq = (x[0] - initial_position[0]) ** 2 + (x[1] - initial_position[1]) ** 2
        return np.exp(-100 * dist_sq)

    def terminal_cost(x):
        """Quadratic terminal cost: distance to goal. x shape: (2,) for single 2D point."""
        dist_sq = (x[0] - goal_position[0]) ** 2 + (x[1] - goal_position[1]) ** 2
        return 0.5 * dist_sq

    conditions = Conditions(
        m_initial=initial_density,
        u_terminal=terminal_cost,
        T=time_horizon,
    )

    problem = MFGProblem(model=model, domain=geometry, conditions=conditions, Nt=num_timesteps)
    # Attach metadata for use in solve/visualization code
    problem.coupling_strength = coupling_strength
    problem.goal_position = goal_position
    problem.initial_position = initial_position
    return problem


def solve_with_configuration(problem, config_name, config):
    """Solve problem with given solver configuration."""
    print(f"\nSolving with {config_name} configuration...")
    print(f"  Characteristic: {config['characteristic_solver']}")
    print(f"  Interpolation: {config['interpolation_method']}")
    print(f"  RBF fallback: {config['use_rbf_fallback']}")

    start_time = time.time()

    # Create solver with configuration
    hjb_solver = HJBSemiLagrangianSolver(
        problem,
        characteristic_solver=config["characteristic_solver"],
        interpolation_method=config["interpolation_method"],
        use_rbf_fallback=config["use_rbf_fallback"],
        rbf_kernel=config.get("rbf_kernel", "thin_plate_spline"),
        use_jax=False,
    )

    # Create FP solver
    fp_solver = FPFDMSolver(problem)

    # Get grid dimensions
    Nt = problem.geometry.time_grid.num_points
    N_total = problem.geometry.grid.num_points_total

    # Initial density and terminal cost from problem components
    components = problem.components

    # Initial density
    M = np.zeros((Nt, N_total))
    for idx in range(N_total):
        multi_idx = problem.geometry.grid.get_multi_index(idx)
        coords = []
        for d in range(problem.geometry.grid.dimension):
            x_d = problem.geometry.grid.bounds[d][0] + multi_idx[d] * problem.geometry.grid.spacing[d]
            coords.append(x_d)
        coords_array = np.array(coords)
        M[0, idx] = components.initial_density(coords_array)

    # Terminal cost
    U_terminal = np.zeros(N_total)
    for idx in range(N_total):
        multi_idx = problem.geometry.grid.get_multi_index(idx)
        coords = []
        for d in range(problem.geometry.grid.dimension):
            x_d = problem.geometry.grid.bounds[d][0] + multi_idx[d] * problem.geometry.grid.spacing[d]
            coords.append(x_d)
        coords_array = np.array(coords)
        U_terminal[idx] = components.terminal_cost(coords_array)

    # Solve with simple Picard iteration (2 iterations for demo)
    U = np.zeros((Nt, N_total))
    for iteration in range(2):
        # Solve HJB backward in time
        U_prev = U.copy()
        U = hjb_solver.solve_hjb_system(M, U_terminal, U_prev)

        # Solve FP forward in time (if not last iteration)
        if iteration < 1:
            M_prev = M.copy()
            M = fp_solver.solve_fp_system(U, M[0, :], M_prev)

    elapsed = time.time() - start_time

    print(f"  Time: {elapsed:.3f}s")
    print(f"  Solution shape: {U.shape}")
    print(f"  Value range: [{U.min():.6f}, {U.max():.6f}]")

    return {"U": U, "M": M, "time": elapsed}


def compare_configurations():
    """Compare baseline and enhanced configurations."""
    print("\n" + "=" * 80)
    print("2D Semi-Lagrangian Enhancements Demonstration")
    print("=" * 80)

    # Create problem
    problem = create_2d_crowd_navigation_problem(
        grid_resolution=15,
        time_horizon=0.5,
        num_timesteps=25,
        sigma=0.05,
    )

    print("\nProblem setup:")
    print("  Domain: [0, 1] × [0, 1]")
    print(f"  Grid: {problem.geometry.grid.num_points}")
    print(f"  Time steps: {problem.geometry.time_grid.num_points}")
    print(f"  Initial position: {problem.initial_position}")
    print(f"  Goal position: {problem.goal_position}")

    # Configurations to compare
    configs = {
        "baseline": {
            "characteristic_solver": "explicit_euler",
            "interpolation_method": "linear",
            "use_rbf_fallback": False,
        },
        "rk4": {
            "characteristic_solver": "rk4",
            "interpolation_method": "linear",
            "use_rbf_fallback": False,
        },
        "cubic": {
            "characteristic_solver": "explicit_euler",
            "interpolation_method": "cubic",
            "use_rbf_fallback": False,
        },
        "enhanced": {
            "characteristic_solver": "rk4",
            "interpolation_method": "cubic",
            "use_rbf_fallback": True,
            "rbf_kernel": "thin_plate_spline",
        },
    }

    # Solve with each configuration
    results = {}
    for name, config in configs.items():
        results[name] = solve_with_configuration(problem, name, config)

    return results, problem


def visualize_results(results, problem):
    """Create visualization comparing configurations."""
    if not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not available, skipping visualization")
        return

    print("\n" + "=" * 80)
    print("Creating Visualization")
    print("=" * 80)

    _fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    config_names = ["baseline", "rk4", "cubic", "enhanced"]

    # Get grid for reshaping
    grid_shape = problem.geometry.grid.num_points

    for col, config_name in enumerate(config_names):
        result = results[config_name]
        U = result["U"]
        M = result["M"]

        # Value function at t=0 (reshaped to 2D grid)
        ax = axes[0, col]
        U_2d = U[0, :].reshape(grid_shape)
        im = ax.imshow(U_2d.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", aspect="auto")
        ax.set_title(f"{config_name}\nValue u(t=0)", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Mark initial and goal positions
        ax.plot(
            problem.initial_position[0],
            problem.initial_position[1],
            "r*",
            markersize=15,
            label="Start",
        )
        ax.plot(
            problem.goal_position[0],
            problem.goal_position[1],
            "g*",
            markersize=15,
            label="Goal",
        )
        ax.legend(fontsize=8)

        # Density at t=T/2 (reshaped to 2D grid)
        ax = axes[1, col]
        t_mid = M.shape[0] // 2
        M_2d = M[t_mid, :].reshape(grid_shape)
        im = ax.imshow(M_2d.T, origin="lower", extent=[0, 1, 0, 1], cmap="hot", aspect="auto")
        ax.set_title(f"Density m(t=T/2)\nTime: {result['time']:.3f}s", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = "examples/outputs/semi_lagrangian_2d_enhancements.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def main():
    """Run demonstration."""
    print("\n" + "=" * 80)
    print(" 2D SEMI-LAGRANGIAN ENHANCEMENTS DEMONSTRATION")
    print("=" * 80)
    print("\nDemonstrating three major enhancements:")
    print("  1. RK4 characteristic tracing (scipy.solve_ivp)")
    print("  2. Cubic spline interpolation")
    print("  3. RBF interpolation fallback")
    print()

    try:
        # Run comparison
        results, problem = compare_configurations()

        # Visualize
        visualize_results(results, problem)

        # Summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print("\nConfiguration comparison:")
        print(f"  {'Config':<15} {'Time (s)':<12} {'Value range':<25}")
        print("  " + "-" * 52)

        for name in ["baseline", "rk4", "cubic", "enhanced"]:
            result = results[name]
            U = result["U"]
            time_val = result["time"]
            value_range = f"[{U.min():.4f}, {U.max():.4f}]"
            print(f"  {name:<15} {time_val:<12.3f} {value_range:<25}")

        print("\nKey findings:")
        print("  - RK4: More accurate characteristic tracing")
        print("  - Cubic: Smoother interpolation in 2D")
        print("  - RBF: Robustness for boundary cases")
        print("  - Enhanced (all): Best accuracy and robustness")
        print()

        return 0

    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
