"""
2D Crowd Motion with FDM Solvers

Demonstrates dimension-agnostic FDM solvers (Phase 2) on a simple 2D crowd
motion problem where agents navigate from starting region to goal region
while minimizing congestion.

Key Features:
- Automatic dimension detection (no 1D vs 2D distinction needed)
- MFGProblem for rectangular domains with nD support
- create_basic_solver() uses HJB-FDM + FP-FDM (both support 2D via dimensional splitting)
- Simple factory interface - dimension handled automatically

Problem Setup:
- Domain: [0,1] × [0,1]
- Initial density: Concentrated at (0.2, 0.2) [bottom-left]
- Goal: (0.8, 0.8) [top-right]
- Agents minimize: travel cost + congestion cost
"""

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.factory import create_basic_solver
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def create_crowd_motion_2d(
    grid_resolution=15,
    time_horizon=0.5,
    num_timesteps=20,
    sigma=0.05,
    congestion_weight=0.5,
    goal=(0.8, 0.8),
    start=(0.2, 0.2),
):
    """
    Create a 2D crowd motion MFG problem.

    Agents navigate from start to goal while avoiding congestion.

    Parameters
    ----------
    grid_resolution : int
        Points per dimension (grid_resolution x grid_resolution)
    time_horizon : float
        Final time T
    num_timesteps : int
        Number of time steps
    sigma : float
        Diffusion coefficient
    congestion_weight : float
        Weight kappa for congestion cost
    goal : tuple
        Goal position (x, y)
    start : tuple
        Initial density center (x, y)

    Returns
    -------
    MFGProblem
        Configured 2D crowd motion problem.
    """
    goal_arr = np.array(goal)
    start_arr = np.array(start)

    # Geometry
    geometry = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[grid_resolution, grid_resolution],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    # Model: H = (1/2)|p|^2 + kappa * m
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: congestion_weight * m,
        coupling_dm=lambda m: congestion_weight,
    )
    model = Model(hamiltonian=hamiltonian, volatility=sigma)

    # Conditions (callables, resolution-independent)
    def initial_density(x):
        """Gaussian blob centered at start. x shape: (2,) for single 2D point."""
        dist_sq = (x[0] - start_arr[0]) ** 2 + (x[1] - start_arr[1]) ** 2
        return np.exp(-100 * dist_sq)

    def terminal_cost(x):
        """Quadratic cost: distance to goal. x shape: (2,) for single 2D point."""
        dist_sq = (x[0] - goal_arr[0]) ** 2 + (x[1] - goal_arr[1]) ** 2
        return 5.0 * dist_sq

    conditions = Conditions(
        m_initial=initial_density,
        u_terminal=terminal_cost,
        T=time_horizon,
    )

    problem = MFGProblem(model=model, domain=geometry, conditions=conditions, Nt=num_timesteps)
    # Attach metadata for use in main()
    problem.congestion_weight = congestion_weight
    problem.goal = goal_arr
    problem.start = start_arr
    return problem


def main():
    """Run 2D crowd motion example."""
    print("\n" + "=" * 70)
    print("  2D Crowd Motion with Dimension-Agnostic FDM Solvers")
    print("=" * 70 + "\n")

    # Create problem
    print("Creating 2D MFG problem...")
    problem = create_crowd_motion_2d(
        grid_resolution=12,  # 12x12 grid
        time_horizon=0.4,
        num_timesteps=15,
        sigma=0.05,
        congestion_weight=0.3,
        goal=(0.8, 0.8),
        start=(0.2, 0.2),
    )

    print("  Domain: [0,1] × [0,1]")
    print(f"  Grid: {problem.geometry.grid.num_points[0] - 1} × {problem.geometry.grid.num_points[1] - 1}")
    print(f"  Timesteps: {problem.Nt + 1}")
    print(f"  Start: {problem.start}")
    print(f"  Goal: {problem.goal}")
    print(f"  Congestion weight κ: {problem.congestion_weight}")

    # Create solver using factory
    # Note: Factory automatically detects 2D and uses dimensional splitting!
    print("\nCreating solver with factory (automatic 2D detection)...")
    solver = create_basic_solver(
        problem,
        damping=0.6,  # Damping for Picard iteration
        max_iterations=20,  # Reduced for faster demo (2D is slow with coarse grid)
        tolerance=1e-3,  # Relaxed tolerance for demonstration
    )

    print(f"  HJB solver: {solver.hjb_solver.__class__.__name__} (dimension={solver.hjb_solver.dimension})")
    print(f"  FP solver: {solver.fp_solver.__class__.__name__} (dimension={solver.fp_solver.dimension})")
    print("  Method: Dimensional splitting (Strang)")

    # Solve
    print("\nSolving MFG system...")
    result = solver.solve()

    # Print results
    print("\n" + "-" * 70)
    print("Results:")
    print("-" * 70)
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.num_iterations}")
    print(f"  Final error: {result.residual:.6e}")

    # Check mass conservation
    U, M = result.U, result.M
    dx, dy = problem.geometry.grid.spacing
    cell_volume = dx * dy
    initial_mass = np.sum(M[0]) * cell_volume
    final_mass = np.sum(M[-1]) * cell_volume
    mass_error = abs(final_mass - initial_mass) / (initial_mass + 1e-10)

    print("\nMass conservation:")
    print(f"  Initial mass: {initial_mass:.6f}")
    print(f"  Final mass: {final_mass:.6f}")
    print(f"  Error: {mass_error:.2%}")

    # Analyze density evolution
    print("\nDensity statistics:")
    print(f"  Initial max: {np.max(M[0]):.6f}")
    print(f"  Final max: {np.max(M[-1]):.6f}")
    print(f"  Initial entropy: {-np.sum(M[0] * np.log(M[0] + 1e-10)) * cell_volume:.4f}")
    print(f"  Final entropy: {-np.sum(M[-1] * np.log(M[-1] + 1e-10)) * cell_volume:.4f}")

    # Analyze value function
    print("\nValue function statistics:")
    print(f"  Initial (at t=0): min={np.min(U[0]):.4f}, max={np.max(U[0]):.4f}")
    print(f"  Final (at t=T): min={np.min(U[-1]):.4f}, max={np.max(U[-1]):.4f}")

    print("\n" + "=" * 70)
    print("  [OK] 2D FDM example complete!")
    print("=" * 70 + "\n")

    print("Key observations:")
    print("  • Factory automatically detected 2D problem")
    print("  • Dimensional splitting used for both HJB and FP")
    print(f"  • Mass error {mass_error:.2%} is typical for FDM splitting ({result.num_iterations} iterations)")
    print("  • No code changes needed vs 1D - dimension-agnostic design works!")

    return result


if __name__ == "__main__":
    main()
