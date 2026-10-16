#!/usr/bin/env python3
"""
Utility Module Demonstration

Demonstrates new utility features from Phase 2.2 improvements:
1. Particle interpolation utilities
2. Geometry utilities (SDF, obstacles)

Run:
    python examples/basic/utility_demo.py
"""

import numpy as np

from mfgarchon.utils import (
    CircleObstacle,
    RectangleObstacle,
    Union,
    create_circle_obstacle,
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
)


def demo_particle_interpolation():
    """Demonstrate particle-grid interpolation utilities."""
    print("\n" + "=" * 60)
    print("Particle Interpolation Demo")
    print("=" * 60)

    # 1D example: Grid to particles
    print("\n1. Grid → Particles (1D)")
    x_grid = np.linspace(0, 1, 50)
    u_grid = np.sin(2 * np.pi * x_grid)  # Smooth function on grid

    particle_positions = np.array([[0.25], [0.5], [0.75]])
    u_particles = interpolate_grid_to_particles(u_grid, ((0, 1),), particle_positions, method="cubic")

    print(f"   Grid points: {len(x_grid)}")
    print(f"   Particles: {len(particle_positions)}")
    print("   Interpolated values at particles:")
    for pos, val in zip(particle_positions, u_particles, strict=True):
        print(f"     x={pos[0]:.2f}: u={val:.3f} (exact: {np.sin(2 * np.pi * pos[0]):.3f})")

    # 2D example: Particles to grid (KDE density estimation)
    print("\n2. Particles → Grid (2D density estimation)")
    particles_2d = np.random.randn(500, 2) * 0.1 + 0.5  # Gaussian blob at (0.5, 0.5)

    # For density estimation from particles (uniform weight = 1.0 for each particle)
    particle_weights = np.ones(len(particles_2d))
    density = interpolate_particles_to_grid(
        particle_values=particle_weights,
        particle_positions=particles_2d,
        grid_shape=(30, 30),
        grid_bounds=((0, 1), (0, 1)),
        method="kde",
    )

    print(f"   Particles: {len(particles_2d)}")
    print(f"   Grid: {density.shape[0]} × {density.shape[1]} = {density.size} points")
    dx = dy = 1.0 / (density.shape[0] - 1)
    print(f"   Density integral: {np.sum(density) * dx * dy:.3f} (should be ≈ 1.0)")
    max_i, max_j = np.unravel_index(np.argmax(density), density.shape)
    max_x = max_i * dx
    max_y = max_j * dy
    print(f"   Max density at: ({max_x:.2f}, {max_y:.2f})")


def demo_geometry_utilities():
    """Demonstrate geometry utility aliases and CSG operations."""
    print("\n" + "=" * 60)
    print("Geometry Utilities Demo")
    print("=" * 60)

    # Create obstacles using convenient aliases
    print("\n1. Creating obstacles")
    wall = RectangleObstacle(np.array([[0.4, 0.6], [0.0, 0.5]]))
    pillar = CircleObstacle(center=np.array([0.7, 0.7]), radius=0.1)

    print("   Wall (rectangle): bounds = [[0.4, 0.6], [0.0, 0.5]]")
    print("   Pillar (circle): center = [0.7, 0.7], radius = 0.1")

    # Alternative: Factory functions
    obstacle3 = create_circle_obstacle(0.3, 0.3, 0.15)
    print("   Obstacle 3 (via factory): center = [0.3, 0.3], radius = 0.15")

    # Combine using CSG operations
    print("\n2. CSG operations (Union)")
    obstacles = Union([wall, pillar, obstacle3])

    # Query signed distance
    test_points = np.array([[0.5, 0.25], [0.1, 0.1], [0.7, 0.7]])

    print("\n3. Signed distance queries")
    for point in test_points:
        sdf = obstacles.signed_distance(point.reshape(1, -1))[0]
        inside = obstacles.contains(point.reshape(1, -1))[0]
        print(f"   Point {point}: SDF = {sdf:+.3f}, inside = {inside}")


def main():
    """Run all utility demonstrations."""
    print("\n" + "=" * 60)
    print("MFGarchon Utility Module Demonstration")
    print("Phase 2.2 Improvements")
    print("=" * 60)

    demo_particle_interpolation()
    demo_geometry_utilities()

    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
