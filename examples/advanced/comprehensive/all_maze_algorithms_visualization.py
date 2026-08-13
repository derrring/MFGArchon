#!/usr/bin/env python3
"""
Comprehensive Maze Algorithm Visualization

Generates high-quality visualizations of all maze generation algorithms
with fine-tuned parameters for detailed comparison.

Saves all visualizations to PNG files (no interactive display).

Author: MFGarchon Team
Date: October 2025
"""

import matplotlib.pyplot as plt
import numpy as np

# Imported from their owner in `geometry/graph/`. The frozen RL package re-exported these and
# was deleted; maze generation was never RL infrastructure, it only had a wrapper there.
from mfgarchon.geometry.graph.maze_cellular_automata import (
    CellularAutomataConfig,
    CellularAutomataGenerator,
)
from mfgarchon.geometry.graph.maze_generator import MazeAlgorithm, MazeGeometry
from mfgarchon.geometry.graph.maze_recursive_division import (
    RecursiveDivisionConfig,
    RecursiveDivisionGenerator,
    add_loops,
    create_room_based_config,
)
from mfgarchon.geometry.graph.maze_voronoi import VoronoiMazeConfig, VoronoiMazeGenerator


def visualize_all_algorithms():
    """Create comprehensive comparison of all maze algorithms."""
    # Disable interactive mode for batch generation
    plt.ioff()

    print("=" * 70)
    print("Generating Comprehensive Maze Algorithm Visualizations")
    print("=" * 70)

    # Use larger sizes for better detail
    size = 60
    seed = 42

    fig, axes = plt.subplots(3, 3, figsize=(20, 20))

    # Row 1: Perfect Mazes
    print("\n1. Generating Perfect Mazes...")

    # Perfect Maze - Recursive Backtracking
    print("   - Recursive Backtracking...")
    perfect_gen = MazeGeometry(size, size, MazeAlgorithm.RECURSIVE_BACKTRACKING)
    perfect_gen.generate(seed=seed)
    perfect_maze = perfect_gen.to_numpy_array()

    axes[0, 0].imshow(perfect_maze, cmap="binary", interpolation="nearest")
    axes[0, 0].set_title("Recursive Backtracking\nPerfect Maze (DFS)", fontweight="bold", fontsize=12)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[0, 0].text(
        0.02,
        0.98,
        f"Size: {size}×{size}\nType: Perfect\nPaths: Long & winding",
        transform=axes[0, 0].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        fontsize=9,
    )

    # Wilson's Algorithm
    print("   - Wilson's Algorithm...")
    wilson_gen = MazeGeometry(size, size, MazeAlgorithm.WILSONS)
    wilson_gen.generate(seed=seed)
    wilson_maze = wilson_gen.to_numpy_array()

    axes[0, 1].imshow(wilson_maze, cmap="binary", interpolation="nearest")
    axes[0, 1].set_title("Wilson's Algorithm\nUnbiased Perfect Maze", fontweight="bold", fontsize=12)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    axes[0, 1].text(
        0.02,
        0.98,
        f"Size: {size}×{size}\nType: Perfect\nDistribution: Unbiased",
        transform=axes[0, 1].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        fontsize=9,
    )

    # Braided Maze (loops added)
    print("   - Braided Maze (with loops)...")
    braided_maze = add_loops(perfect_maze, loop_density=0.2, seed=seed)

    axes[0, 2].imshow(braided_maze, cmap="binary", interpolation="nearest")
    axes[0, 2].set_title("Braided Maze\nPerfect Maze + Loops", fontweight="bold", fontsize=12)
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    added = np.sum(braided_maze == 0) - np.sum(perfect_maze == 0)
    axes[0, 2].text(
        0.02,
        0.98,
        f"Size: {size}×{size}\nLoop density: 20%\nAdded paths: {added}",
        transform=axes[0, 2].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        fontsize=9,
    )

    # Row 2: Room-Based Mazes
    print("\n2. Generating Room-Based Mazes...")

    # Recursive Division - Small Rooms
    print("   - Recursive Division (small rooms)...")
    rd_small_config = create_room_based_config(size, size, room_size="small", corridor_width="narrow", seed=seed)
    rd_small_gen = RecursiveDivisionGenerator(rd_small_config)
    rd_small_maze = rd_small_gen.generate()

    axes[1, 0].imshow(rd_small_maze, cmap="binary", interpolation="nearest")
    axes[1, 0].set_title("Recursive Division\nSmall Rooms, Narrow Corridors", fontweight="bold", fontsize=12)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    open_pct = 100 * np.sum(rd_small_maze == 0) / rd_small_maze.size
    axes[1, 0].text(
        0.02,
        0.98,
        f"Size: {size}×{size}\nRoom: 3×3 min\nOpen: {open_pct:.1f}%",
        transform=axes[1, 0].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        fontsize=9,
    )

    # Recursive Division - Large Rooms
    print("   - Recursive Division (large rooms)...")
    rd_large_config = create_room_based_config(size, size, room_size="large", corridor_width="wide", seed=seed)
    rd_large_gen = RecursiveDivisionGenerator(rd_large_config)
    rd_large_maze = rd_large_gen.generate()

    axes[1, 1].imshow(rd_large_maze, cmap="binary", interpolation="nearest")
    axes[1, 1].set_title("Recursive Division\nLarge Rooms, Wide Corridors", fontweight="bold", fontsize=12)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    open_pct = 100 * np.sum(rd_large_maze == 0) / rd_large_maze.size
    axes[1, 1].text(
        0.02,
        0.98,
        f"Size: {size}×{size}\nRoom: 8×8 min\nOpen: {open_pct:.1f}%",
        transform=axes[1, 1].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        fontsize=9,
    )

    # Recursive Division with loops
    print("   - Recursive Division (with loops)...")
    rd_braided = add_loops(rd_large_maze, loop_density=0.15, seed=seed)

    axes[1, 2].imshow(rd_braided, cmap="binary", interpolation="nearest")
    axes[1, 2].set_title("Recursive Division + Loops\nMultiple Path Options", fontweight="bold", fontsize=12)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    axes[1, 2].text(
        0.02,
        0.98,
        f"Size: {size}×{size}\nLoop density: 15%\nRoute diversity",
        transform=axes[1, 2].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        fontsize=9,
    )

    # Row 3: Organic Mazes
    print("\n3. Generating Organic Mazes...")

    # Cellular Automata
    print("   - Cellular Automata...")
    ca_config = CellularAutomataConfig(
        rows=size, cols=size, initial_wall_prob=0.45, num_iterations=5, ensure_connectivity=True, seed=seed
    )
    ca_gen = CellularAutomataGenerator(ca_config)
    ca_maze = ca_gen.generate()

    axes[2, 0].imshow(ca_maze, cmap="binary", interpolation="nearest")
    axes[2, 0].set_title("Cellular Automata\nOrganic, Cave-like", fontweight="bold", fontsize=12)
    axes[2, 0].set_xticks([])
    axes[2, 0].set_yticks([])
    open_pct = 100 * np.sum(ca_maze == 0) / ca_maze.size
    axes[2, 0].text(
        0.02,
        0.98,
        f"Size: {size}×{size}\nIterations: 5\nOpen: {open_pct:.1f}%",
        transform=axes[2, 0].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        fontsize=9,
    )

    # Voronoi - Sparse
    print("   - Voronoi (sparse, few rooms)...")
    voronoi_sparse_config = VoronoiMazeConfig(
        rows=size, cols=size, num_points=10, relaxation_iterations=0, wall_thickness=1, seed=seed
    )
    voronoi_sparse_gen = VoronoiMazeGenerator(voronoi_sparse_config)
    voronoi_sparse_maze = voronoi_sparse_gen.generate()

    axes[2, 1].imshow(voronoi_sparse_maze, cmap="binary", interpolation="nearest")
    axes[2, 1].set_title("Voronoi Diagram\nSparse, Large Rooms", fontweight="bold", fontsize=12)
    axes[2, 1].set_xticks([])
    axes[2, 1].set_yticks([])
    # Overlay seed points
    if voronoi_sparse_gen.points is not None:
        axes[2, 1].scatter(
            voronoi_sparse_gen.points[:, 0], voronoi_sparse_gen.points[:, 1], c="red", s=40, marker="x", alpha=0.8
        )
    open_pct = 100 * np.sum(voronoi_sparse_maze == 0) / voronoi_sparse_maze.size
    axes[2, 1].text(
        0.02,
        0.98,
        f"Size: {size}×{size}\nRooms: 10\nOpen: {open_pct:.1f}%",
        transform=axes[2, 1].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        fontsize=9,
    )

    # Voronoi - Dense with relaxation
    print("   - Voronoi (dense, relaxed)...")
    voronoi_dense_config = VoronoiMazeConfig(
        rows=size, cols=size, num_points=20, relaxation_iterations=3, wall_thickness=1, seed=seed
    )
    voronoi_dense_gen = VoronoiMazeGenerator(voronoi_dense_config)
    voronoi_dense_maze = voronoi_dense_gen.generate()

    axes[2, 2].imshow(voronoi_dense_maze, cmap="binary", interpolation="nearest")
    axes[2, 2].set_title("Voronoi Diagram\nDense, Uniform (Relaxed)", fontweight="bold", fontsize=12)
    axes[2, 2].set_xticks([])
    axes[2, 2].set_yticks([])
    # Overlay seed points
    if voronoi_dense_gen.points is not None:
        axes[2, 2].scatter(
            voronoi_dense_gen.points[:, 0], voronoi_dense_gen.points[:, 1], c="red", s=25, marker="o", alpha=0.7
        )
    open_pct = 100 * np.sum(voronoi_dense_maze == 0) / voronoi_dense_maze.size
    axes[2, 2].text(
        0.02,
        0.98,
        f"Size: {size}×{size}\nRooms: 20\nRelax: 3 iter",
        transform=axes[2, 2].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        fontsize=9,
    )

    fig.suptitle("Complete Maze Algorithm Portfolio for MFG Research", fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    filename = "complete_maze_algorithm_portfolio.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved: {filename}")
    plt.close()


def visualize_algorithm_characteristics():
    """Visualize specific characteristics of each algorithm."""
    print("\n" + "=" * 70)
    print("Generating Algorithm Characteristics Comparison")
    print("=" * 70)

    size = 50
    seed = 42

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # Recursive Backtracking - show long paths
    print("\n1. Perfect Maze Characteristics...")
    perfect_gen = MazeGeometry(size, size, MazeAlgorithm.RECURSIVE_BACKTRACKING)
    perfect_gen.generate(seed=seed)
    perfect_maze = perfect_gen.to_numpy_array()

    axes[0, 0].imshow(perfect_maze, cmap="binary", interpolation="nearest")
    axes[0, 0].set_title("Recursive Backtracking\nLong Winding Paths", fontweight="bold")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # Wilson's - show unbiased
    wilson_gen = MazeGeometry(size, size, MazeAlgorithm.WILSONS)
    wilson_gen.generate(seed=seed)
    wilson_maze = wilson_gen.to_numpy_array()

    axes[0, 1].imshow(wilson_maze, cmap="binary", interpolation="nearest")
    axes[0, 1].set_title("Wilson's Algorithm\nUnbiased Distribution", fontweight="bold")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # Recursive Division - structured
    print("2. Room-Based Characteristics...")
    rd_config = RecursiveDivisionConfig(rows=size, cols=size, min_room_width=6, min_room_height=6, seed=seed)
    rd_gen = RecursiveDivisionGenerator(rd_config)
    rd_maze = rd_gen.generate()

    axes[0, 2].imshow(rd_maze, cmap="binary", interpolation="nearest")
    axes[0, 2].set_title("Recursive Division\nStructured Rooms", fontweight="bold")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])

    # Cellular Automata - organic
    print("3. Organic Characteristics...")
    ca_config = CellularAutomataConfig(rows=size, cols=size, initial_wall_prob=0.45, num_iterations=6, seed=seed)
    ca_gen = CellularAutomataGenerator(ca_config)
    ca_maze = ca_gen.generate()

    axes[0, 3].imshow(ca_maze, cmap="binary", interpolation="nearest")
    axes[0, 3].set_title("Cellular Automata\nOrganic Growth", fontweight="bold")
    axes[0, 3].set_xticks([])
    axes[0, 3].set_yticks([])

    # Voronoi - natural rooms
    print("4. Voronoi Characteristics...")
    voronoi_config = VoronoiMazeConfig(rows=size, cols=size, num_points=15, relaxation_iterations=2, seed=seed)
    voronoi_gen = VoronoiMazeGenerator(voronoi_config)
    voronoi_maze = voronoi_gen.generate()

    axes[1, 0].imshow(voronoi_maze, cmap="binary", interpolation="nearest")
    axes[1, 0].set_title("Voronoi\nIrregular Rooms", fontweight="bold")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    if voronoi_gen.points is not None:
        axes[1, 0].scatter(voronoi_gen.points[:, 0], voronoi_gen.points[:, 1], c="red", s=30, marker="o", alpha=0.7)

    # Different densities
    print("5. Density Variations...")

    # Sparse Voronoi
    voronoi_sparse = VoronoiMazeGenerator(VoronoiMazeConfig(rows=size, cols=size, num_points=8, seed=seed)).generate()
    axes[1, 1].imshow(voronoi_sparse, cmap="binary", interpolation="nearest")
    axes[1, 1].set_title("Voronoi Sparse\n8 Large Rooms", fontweight="bold")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    # Dense Voronoi
    voronoi_dense = VoronoiMazeGenerator(
        VoronoiMazeConfig(rows=size, cols=size, num_points=25, relaxation_iterations=3, seed=seed)
    ).generate()
    axes[1, 2].imshow(voronoi_dense, cmap="binary", interpolation="nearest")
    axes[1, 2].set_title("Voronoi Dense\n25 Small Rooms", fontweight="bold")
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    # Braided comparison
    print("6. Loop Density Effects...")
    braided_heavy = add_loops(perfect_maze, loop_density=0.3, seed=seed)
    axes[1, 3].imshow(braided_heavy, cmap="binary", interpolation="nearest")
    axes[1, 3].set_title("Heavy Braiding\n30% Loop Density", fontweight="bold")
    axes[1, 3].set_xticks([])
    axes[1, 3].set_yticks([])

    fig.suptitle("Algorithm Characteristics: Detailed Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    filename = "algorithm_characteristics.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved: {filename}")
    plt.close()


def visualize_mfg_scenarios():
    """Generate realistic MFG scenario visualizations."""
    print("\n" + "=" * 70)
    print("Generating MFG Application Scenarios")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    scenarios = [
        # (name, generator_type, config, description)
        (
            "Office Building",
            "recursive_division",
            {"rows": 70, "cols": 70, "min_room_width": 5, "min_room_height": 5, "seed": 42},
            "Structured office layout",
        ),
        (
            "Museum",
            "voronoi",
            {"rows": 70, "cols": 70, "num_points": 12, "relaxation_iterations": 2, "seed": 42},
            "Irregular gallery spaces",
        ),
        (
            "Shopping Mall",
            "voronoi",
            {"rows": 70, "cols": 70, "num_points": 18, "relaxation_iterations": 3, "seed": 42},
            "Multiple shop spaces",
        ),
        (
            "Hospital",
            "recursive_division",
            {"rows": 70, "cols": 70, "min_room_width": 7, "min_room_height": 7, "door_width": 3, "seed": 42},
            "Large wards + corridors",
        ),
        (
            "Park / Plaza",
            "cellular_automata",
            {"rows": 70, "cols": 70, "initial_wall_prob": 0.42, "num_iterations": 6, "seed": 42},
            "Organic open spaces",
        ),
        (
            "Conference Center",
            "voronoi",
            {"rows": 70, "cols": 70, "num_points": 15, "relaxation_iterations": 4, "wall_thickness": 2, "seed": 42},
            "Multiple meeting rooms",
        ),
    ]

    for idx, (name, gen_type, config, description) in enumerate(scenarios):
        row = idx // 3
        col = idx % 3

        print(f"\n{idx + 1}. Generating {name}...")

        if gen_type == "recursive_division":
            generator = RecursiveDivisionGenerator(RecursiveDivisionConfig(**config))
        elif gen_type == "voronoi":
            generator = VoronoiMazeGenerator(VoronoiMazeConfig(**config))
        elif gen_type == "cellular_automata":
            generator = CellularAutomataGenerator(CellularAutomataConfig(**config))

        maze = generator.generate()

        axes[row, col].imshow(maze, cmap="binary", interpolation="nearest")
        axes[row, col].set_title(f"{name}\n{description}", fontweight="bold", fontsize=11)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

        open_pct = 100 * np.sum(maze == 0) / maze.size
        info = f"Open: {open_pct:.1f}%\nAlgo: {gen_type.replace('_', ' ').title()}"
        axes[row, col].text(
            0.02,
            0.98,
            info,
            transform=axes[row, col].transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
            fontsize=9,
        )

    fig.suptitle("Realistic MFG Application Scenarios", fontsize=16, fontweight="bold")
    plt.tight_layout()

    filename = "mfg_realistic_scenarios.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved: {filename}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("COMPREHENSIVE MAZE ALGORITHM VISUALIZATION")
    print("=" * 70)
    print("\nGenerating high-resolution visualizations...")
    print("This may take 1-2 minutes for quality rendering.\n")

    visualize_all_algorithms()
    visualize_algorithm_characteristics()
    visualize_mfg_scenarios()

    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  1. complete_maze_algorithm_portfolio.png (3×3 grid, all algorithms)")
    print("  2. algorithm_characteristics.png (2×4 grid, detailed features)")
    print("  3. mfg_realistic_scenarios.png (2×3 grid, application scenarios)")
    print("\nAll images saved at 300 DPI for publication quality.")
    print("=" * 70)


if __name__ == "__main__":
    main()
