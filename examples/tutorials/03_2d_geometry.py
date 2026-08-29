"""
Tutorial 03: 2D Geometry

Learn how to work with 2D spatial domains in MFGarchon.

What you'll learn:
- How to create 2D problems with TensorProductGrid
- How to define 2D initial density and terminal cost
- How to visualize 2D density evolution
- How mass conservation works in 2D

Mathematical Problem:
    2D crowd navigation with target attraction:
    - Domain: [0,10] x [0,10] (square room)
    - Target: Center at (5,5) - agents want to reach it
    - Hamiltonian: H = (1/2)|p|^2 + lambda*m (LQ with congestion)
    - Initial: Agents start at corners
"""

import numpy as np

from mfgarchon import Conditions, MFGProblem, Model
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

if __name__ == "__main__":
    # ==============================================================================
    # Step 1: Create 2D Problem
    # ==============================================================================

    print("=" * 70)
    print("TUTORIAL 03: 2D Geometry")
    print("=" * 70)
    print()

    # Problem parameters (reduced resolution for faster demo)
    TARGET = np.array([5.0, 5.0])  # Target location (center of room)
    CONGESTION_WEIGHT = 1.0
    GRID_RESOLUTION = 15  # Reduced from 30 for faster execution
    SIGMA = 0.3  # Slightly higher for stability

    # Create 2D geometry
    print("Creating 2D geometry...")
    geometry = TensorProductGrid(
        bounds=[(0.0, 10.0), (0.0, 10.0)],  # [(xmin, xmax), (ymin, ymax)]
        Nx_points=[GRID_RESOLUTION, GRID_RESOLUTION],  # 30x30 spatial grid
        boundary_conditions=no_flux_bc(dimension=2),
    )

    print(f"  Domain: {geometry.get_bounds()}")
    print(f"  Grid: {GRID_RESOLUTION} x {GRID_RESOLUTION}")
    print()

    # Corner positions for initial density
    corners = [
        (2.0, 2.0),  # Bottom-left
        (8.0, 2.0),  # Bottom-right
        (2.0, 8.0),  # Top-left
        (8.0, 8.0),  # Top-right
    ]

    # Define Model (game rules)
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: CONGESTION_WEIGHT * m,
        coupling_dm=lambda m: CONGESTION_WEIGHT,
    )
    model = Model(hamiltonian=hamiltonian, sigma=SIGMA)

    # Define Conditions (callables for resolution-independence)
    def terminal_cost(x):
        """Distance to target. x shape: (2,) for a single 2D point."""
        return (x[0] - TARGET[0]) ** 2 + (x[1] - TARGET[1]) ** 2

    def initial_density(x):
        """4 Gaussian blobs at corners. x shape: (2,) for a single 2D point."""
        m = 0.0
        for cx, cy in corners:
            m += np.exp(-5 * ((x[0] - cx) ** 2 + (x[1] - cy) ** 2))
        return m

    conditions = Conditions(
        u_terminal=terminal_cost,
        # `initial_density` sums four Gaussian blobs and integrates to 2.578562 on this
        # 15x15 grid over [0,10]^2 -- ABOVE 1, unlike the 1-D tutorials. The library measures
        # the mass, names the measure, and warns, but does not rescale it (#1887). In 2-D the
        # over-unit case is easy to reach by accident, which is exactly why the warning
        # exists: it is the difference between a deliberate choice and a miscoded density.
        m_initial=initial_density,
        T=2.0,
    )

    print("Setting up initial density (4 Gaussian blobs at corners)...")
    print()

    # Create problem
    print("Creating 2D target attraction problem...")
    problem = MFGProblem(model=model, domain=geometry, conditions=conditions, Nt=20)

    print(f"  Target: {TARGET}")
    print(f"  Time horizon: T = {problem.T}")
    print(f"  Diffusion: sigma = {problem.sigma}")
    print()

    # ==============================================================================
    # Step 2: Solve the Problem
    # ==============================================================================

    print("Solving 2D MFG system...")
    print("(This may take a moment - 2D problems are more computationally intensive)")
    print()

    result = problem.solve(verbose=True)

    print()
    print(f"Converged: {result.converged} (iterations: {result.iterations})")
    print()

    # ==============================================================================
    # Step 3: Mass Conservation Check
    # ==============================================================================

    print("=" * 70)
    print("MASS CONSERVATION (2D)")
    print("=" * 70)
    print()

    # Get grid info
    grid_shape = geometry.get_grid_shape()
    dx, dy = geometry.get_grid_spacing()

    # Check mass at each timestep using trapezoidal integration
    masses = []
    for t_idx in range(result.M.shape[0]):
        m_2d = result.M[t_idx].reshape(grid_shape)
        mass = np.trapezoid(np.trapezoid(m_2d, dx=dy, axis=1), dx=dx)
        masses.append(mass)

    print(f"Initial mass: {masses[0]:.6f}")
    print(f"Final mass:   {masses[-1]:.6f}")
    print(f"Mass drift:   {abs(masses[-1] - masses[0]):.6e}")
    print()

    # ==============================================================================
    # Step 4: Visualize 2D Solution
    # ==============================================================================

    try:
        from pathlib import Path

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(15, 10))

        # Create spatial grids for plotting
        x_1d = np.linspace(0, 10, GRID_RESOLUTION)
        y_1d = np.linspace(0, 10, GRID_RESOLUTION)
        X, Y = np.meshgrid(x_1d, y_1d, indexing="ij")

        # Time points to plot
        t_space = np.linspace(0, problem.T, problem.Nt + 1)
        times_to_plot = [0, len(t_space) // 3, 2 * len(t_space) // 3, -1]

        for idx, t_idx in enumerate(times_to_plot, 1):
            ax = fig.add_subplot(2, 2, idx)

            m_2d = result.M[t_idx].reshape(grid_shape)

            # Density heatmap
            contour = ax.contourf(X, Y, m_2d, levels=20, cmap="viridis")
            ax.plot(*TARGET, "r*", markersize=20, label="Target")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Density at t={t_space[t_idx]:.2f}s")
            ax.set_aspect("equal")
            plt.colorbar(contour, ax=ax)
            ax.legend()

        plt.tight_layout()

        # Save plot
        output_dir = Path(__file__).parent.parent / "outputs" / "tutorials"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "03_2d_geometry.png"
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
    print("  1. How to create 2D problems with TensorProductGrid")
    print("  2. How to define 2D initial density and terminal cost functions")
    print("  3. How to compute 2D integrals for mass conservation")
    print("  4. How to visualize 2D density evolution")
    print()
    print("Key API elements for 2D:")
    print("  - Model(hamiltonian=..., sigma=...)")
    print("  - TensorProductGrid(bounds=[(xmin,xmax), (ymin,ymax)], Nx_points=[Nx, Ny])")
    print("  - Conditions with callables: f(x) where x is shape (2,) for 2D points")
    print("  - boundary_conditions=no_flux_bc(dimension=2)")
    print()
    print("Next: Tutorial 04 - Advanced Solver Configuration")
    print("=" * 70)
