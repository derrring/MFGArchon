"""
Complete example: Capacity-constrained MFG for maze navigation.

This example demonstrates:
1. Maze generation with structured corridors
2. Capacity field computation from geometry
3. Capacity-constrained MFG problem setup
4. Solving the coupled HJB-FP system
5. Visualization of results

Mathematical Framework:
    Extended Hamiltonian with congestion term:
        H(x, m, ∇u) = (1/2)|∇u|² + α·m + γ·g(m/C)

    where:
    - m(x): agent density
    - C(x): corridor capacity (from distance transform)
    - g(ρ): convex congestion cost (ρ = m/C)
    - γ: congestion weight parameter

As density approaches capacity, g(m/C) → ∞ creates a "soft wall" effect,
encouraging agents to seek alternative routes.

Usage:
    python example_maze_mfg.py [--size 20] [--congestion quadratic] [--gamma 1.0]

Created: 2025-11-12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.advanced.capacity_constrained_mfg import (
    CapacityConstrainedMFGProblem,
    CapacityField,
    create_congestion_model,
    visualize_capacity_field,
)
from mfgarchon.geometry.graph import MazeConfig, MazeGeometry


def generate_maze_with_capacity(
    rows: int = 20,
    cols: int = 20,
    wall_thickness: int = 3,
    epsilon: float = 1e-3,
) -> tuple[np.ndarray, CapacityField]:
    """
    Generate maze and compute capacity field.

    Args:
        rows: Number of maze rows
        cols: Number of maze columns
        wall_thickness: Thickness of maze walls (pixels)
        epsilon: Minimum capacity threshold

    Returns:
        maze_array: Binary maze (1=wall, 0=passage)
        capacity: CapacityField instance
    """
    print(f"\n1. Generating {rows}x{cols} maze with wall_thickness={wall_thickness}...")

    # Generate perfect maze
    config = MazeConfig(rows=rows, cols=cols, seed=42)
    maze = MazeGeometry.generate(config)
    maze_array = maze.to_numpy_array(wall_thickness=wall_thickness)

    print(f"   Maze shape: {maze_array.shape}")
    print(f"   Passages: {np.sum(maze_array == 0)} cells ({100 * np.mean(maze_array == 0):.1f}%)")
    print(f"   Walls: {np.sum(maze_array == 1)} cells ({100 * np.mean(maze_array == 1):.1f}%)")

    # Compute capacity field from geometry
    print(f"\n2. Computing capacity field (epsilon={epsilon})...")
    capacity = CapacityField.from_maze_geometry(
        maze_array,
        wall_thickness=wall_thickness,
        epsilon=epsilon,
        normalization="max",
        decay_factor=0.5,  # Smooth falloff near walls
    )

    print(f"   Capacity range: [{capacity.min_capacity:.3f}, {capacity.max_capacity:.3f}]")
    print(f"   Mean capacity: {capacity.mean_capacity:.3f}")

    return maze_array, capacity


def create_capacity_constrained_problem(
    capacity: CapacityField,
    congestion_type: str = "quadratic",
    congestion_weight: float = 1.0,
    spatial_bounds: tuple[tuple[float, float], ...] = ((0, 1), (0, 1)),
    Nx: int = 63,
    Ny: int = 63,
    T: float = 1.0,
    Nt: int = 50,
    sigma: float = 0.01,
) -> CapacityConstrainedMFGProblem:
    """
    Create capacity-constrained MFG problem.

    Args:
        capacity: CapacityField instance
        congestion_type: Type of congestion model ("quadratic", "exponential", "logbarrier")
        congestion_weight: Weight γ for congestion term
        spatial_bounds: Physical domain bounds
        Nx, Ny: Spatial discretization
        T: Terminal time
        Nt: Time discretization
        sigma: Diffusion coefficient

    Returns:
        CapacityConstrainedMFGProblem instance
    """
    print("\n3. Creating capacity-constrained MFG problem...")
    print(f"   Congestion model: {congestion_type}")
    print(f"   Congestion weight γ: {congestion_weight}")
    print(f"   Grid: {Nx}x{Ny}, T={T}, Nt={Nt}, σ={sigma}")

    # Create congestion model
    congestion_model = create_congestion_model(congestion_type)

    # Create problem
    problem = CapacityConstrainedMFGProblem(
        capacity_field=capacity,
        congestion_model=congestion_model,
        congestion_weight=congestion_weight,
        spatial_bounds=spatial_bounds,
        spatial_discretization=[Nx, Ny],
        T=T,
        Nt=Nt,
        volatility=sigma,
    )

    print(f"   Problem created: dimension={problem.dimension}")

    return problem


def solve_and_visualize(
    problem: CapacityConstrainedMFGProblem,
    maze_array: np.ndarray,
    capacity: CapacityField,
    output_dir: Path | None = None,
) -> None:
    """
    Solve MFG and visualize results.

    Args:
        problem: CapacityConstrainedMFGProblem instance
        maze_array: Binary maze for overlay
        capacity: CapacityField for visualization
        output_dir: Optional output directory for saving figures
    """
    print("\n4. Solving capacity-constrained MFG system...")
    print("   Note: This is a placeholder - full solver integration pending")

    # TODO: Integrate with MFGarchon solvers
    # from mfgarchon.factory import create_fast_solver
    # solver = create_fast_solver(problem)
    # result = solver.solve()

    # For now, create synthetic data for visualization demo
    print("   Creating synthetic visualization data...")
    Nx, Ny = problem.geometry.Nx_points

    # Synthetic density (concentrated in corridors)
    m_synthetic = np.random.rand(Nx, Ny) * 0.5
    m_synthetic[maze_array[:Nx, :Ny] == 1] = 0.0  # Zero density in walls

    # Synthetic value function (smooth)
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)
    u_synthetic = np.sin(np.pi * X) * np.sin(np.pi * Y)

    print("\n5. Visualizing results...")

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Capacity-Constrained MFG: Maze Navigation", fontsize=16, fontweight="bold")

    # Row 1: Geometry and Capacity
    # (a) Maze geometry
    axes[0, 0].imshow(maze_array.T, origin="lower", cmap="binary", interpolation="nearest")
    axes[0, 0].set_title("(a) Maze Geometry")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")

    # (b) Capacity field C(x)
    im1 = axes[0, 1].imshow(capacity.capacity.T, origin="lower", cmap="viridis", interpolation="nearest")
    axes[0, 1].set_title("(b) Capacity Field C(x)")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0, 1], label="Corridor Capacity")

    # (c) Capacity histogram
    axes[0, 2].hist(capacity.capacity.flatten(), bins=50, alpha=0.7, edgecolor="black")
    axes[0, 2].set_title("(c) Capacity Distribution")
    axes[0, 2].set_xlabel("Capacity C(x)")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].axvline(capacity.epsilon, color="red", linestyle="--", label=f"ε={capacity.epsilon}")
    axes[0, 2].legend()

    # Row 2: MFG Solution (synthetic data)
    # (d) Agent density m(t,x)
    im2 = axes[1, 0].imshow(m_synthetic.T, origin="lower", cmap="hot", interpolation="bilinear")
    axes[1, 0].set_title("(d) Agent Density m(t,x) [Synthetic]")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    plt.colorbar(im2, ax=axes[1, 0], label="Density")

    # (e) Value function u(t,x)
    im3 = axes[1, 1].imshow(u_synthetic.T, origin="lower", cmap="coolwarm", interpolation="bilinear")
    axes[1, 1].set_title("(e) Value Function u(t,x) [Synthetic]")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    plt.colorbar(im3, ax=axes[1, 1], label="Value")

    # (f) Congestion ratio ρ(x) = m(x)/C(x)
    # Interpolate capacity to match m_synthetic grid
    positions = np.array([[i, j] for i in range(Nx) for j in range(Ny)])
    capacity_interp = capacity.interpolate_at_positions(positions).reshape(Nx, Ny)
    congestion_ratio = m_synthetic / (capacity_interp + 1e-6)

    im4 = axes[1, 2].imshow(
        congestion_ratio.T, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1.5, interpolation="bilinear"
    )
    axes[1, 2].set_title("(f) Congestion Ratio ρ(x)=m/C [Synthetic]")
    axes[1, 2].set_xlabel("x")
    axes[1, 2].set_ylabel("y")
    cbar = plt.colorbar(im4, ax=axes[1, 2], label="Congestion Ratio")
    cbar.ax.axhline(1.0, color="black", linestyle="--", linewidth=2)

    plt.tight_layout()

    # Save if output directory provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "maze_mfg_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"   Saved figure to: {output_path}")

    plt.show()

    print("\n6. Summary Statistics:")
    print(f"   Max density: {np.max(m_synthetic):.3f}")
    print(f"   Mean density (corridors): {np.mean(m_synthetic[m_synthetic > 0]):.3f}")
    print(f"   Max congestion ratio: {np.max(congestion_ratio):.3f}")
    print(
        f"   Overcapacity cells (ρ>1): {np.sum(congestion_ratio > 1.0)} ({100 * np.mean(congestion_ratio > 1.0):.1f}%)"
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Capacity-constrained MFG for maze navigation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", type=int, default=20, help="Maze size (rows and cols)")
    parser.add_argument(
        "--congestion",
        type=str,
        default="quadratic",
        choices=["quadratic", "exponential", "logbarrier"],
        help="Congestion model type",
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="Congestion weight parameter")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for figures")

    args = parser.parse_args()

    print("=" * 80)
    print("Capacity-Constrained MFG: Maze Navigation Example")
    print("=" * 80)

    # Generate maze and capacity field
    maze_array, capacity = generate_maze_with_capacity(
        rows=args.size,
        cols=args.size,
        wall_thickness=3,
        epsilon=1e-3,
    )

    # Visualize capacity field
    print("\n   Displaying capacity field visualization...")
    visualize_capacity_field(capacity, maze_array, figsize=(14, 5))

    # Create capacity-constrained problem
    problem = create_capacity_constrained_problem(
        capacity=capacity,
        congestion_type=args.congestion,
        congestion_weight=args.gamma,
        Nx=63,
        Ny=63,
        T=1.0,
        Nt=50,
        sigma=0.01,
    )

    # Solve and visualize
    output_dir = Path(args.output_dir) if args.output_dir else None
    solve_and_visualize(problem, maze_array, capacity, output_dir)

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
