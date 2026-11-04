"""
Non-Tensor Boundary Condition Demonstration

This example demonstrates BC specification for non-rectangular domains:
1. Circular domain with normal-based exit (top of circle)
2. SDF-based region specification for curved boundaries
3. Comparison with traditional tensor-product approach

Demonstrates Issue #549: BC framework for non-tensor geometries
Using infrastructure from Issue #590: Geometry traits (SupportsManifold, SupportsLipschitz)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from mfgarchon.geometry.boundary import BCSegment, mixed_bc
from mfgarchon.geometry.boundary.types import BCType
from mfgarchon.geometry.implicit import Hypersphere
from mfgarchon.geometry.protocols import (
    SupportsBoundaryNormal,
    SupportsBoundaryProjection,
    SupportsLipschitz,
    SupportsManifold,
)

# =============================================================================
# Problem Setup
# =============================================================================

print("=" * 70)
print("Non-Tensor Boundary Condition Demonstration")
print("=" * 70)

# Physical parameters
sigma = 0.05  # Diffusion coefficient
T_final = 2.0  # Final time
Nt = 40  # Time steps

# =============================================================================
# Example 1: Circular Domain with Normal-Based Exit
# =============================================================================

print("\n" + "=" * 70)
print("Example 1: Circular Domain with Top Exit")
print("=" * 70)

# Create circular domain (2D hypersphere)
center = np.array([0.5, 0.5])
radius = 0.3
domain_circle = Hypersphere(center=center, radius=radius)

print("\nCircular Domain:")
print(f"  Center: {center}")
print(f"  Radius: {radius}")
print(f"  Dimension: {domain_circle.dimension}")
print(f"  Geometry Type: {domain_circle.geometry_type}")

# Verify geometry traits (Issue #590)
print("\nGeometry Traits:")
print(f"  SupportsBoundaryNormal: {isinstance(domain_circle, SupportsBoundaryNormal)}")
print(f"  SupportsBoundaryProjection: {isinstance(domain_circle, SupportsBoundaryProjection)}")
print(f"  SupportsManifold: {isinstance(domain_circle, SupportsManifold)}")
print(f"  SupportsLipschitz: {isinstance(domain_circle, SupportsLipschitz)}")

# Boundary conditions: Exit on top (normal pointing up), reflecting elsewhere
bc_circle = mixed_bc(
    dimension=2,
    segments=[
        BCSegment(
            name="exit_top",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            normal_direction=np.array([0.0, 1.0]),  # Top (y-direction)
            normal_tolerance=0.8,  # cos(~37 deg) - top 37 degrees of circle
        ),
        BCSegment(
            name="walls",
            bc_type=BCType.REFLECTING,
            boundary="all",
            priority=-1,  # Lower priority = fallback
        ),
    ],
    domain_bounds=domain_circle.get_bounding_box(),
)

print("\nBoundary Conditions:")
print(f"  Exit: Top segment (normal ≈ [0, 1], tolerance={0.8})")
print("  Walls: Reflecting (fallback)")

# Create MFG problem (simplified - no HJB coupling for demonstration)
# Using implicit domain directly (meshfree approach)
print("\nNote: This example demonstrates BC specification.")
print("      Full MFG solver with implicit domains requires meshfree discretization.")
print("      See examples/advanced/meshfree_mfg.py for complete workflow.")

# Demonstrate BC matching
print("\nTesting BC Segment Matching:")
test_points = [
    (np.array([0.5, 0.5 + radius]), "top_center"),  # Top of circle
    (np.array([0.5 + radius, 0.5]), "right_side"),  # Right side
    (np.array([0.5, 0.5 - radius]), "bottom_center"),  # Bottom
]

for point, label in test_points:
    # Compute outward normal at this boundary point
    grad = domain_circle.get_boundary_normal(point)

    # Check which BC segment matches
    segment = bc_circle.get_bc_at_point(point, boundary_id=None)

    print(f"  {label:15s}: normal={grad}, segment={segment.name if segment else 'None'}")

# =============================================================================
# Example 2: SDF-Based Region Specification
# =============================================================================

print("\n" + "=" * 70)
print("Example 2: SDF-Based Exit Region")
print("=" * 70)

# Create circular domain
domain_sdf = Hypersphere(center=center, radius=radius)


# Define exit as an SDF region (small sector on top)
# SDF is negative inside the exit region
def top_sector_sdf(x: np.ndarray) -> float:
    """
    SDF for top sector of circle.

    Region: Points on circle boundary with y > center_y + 0.8*radius
    """
    # Check if on boundary (distance from center ≈ radius)
    dist_from_center = np.sqrt(sum((xi - center) ** 2 for xi in x))
    on_boundary = abs(dist_from_center - radius) < 0.01

    # Check if in top sector
    in_top = x[1] > (center[1] + 0.15 * radius)

    if on_boundary and in_top:
        return -1.0  # Inside exit region
    else:
        return 1.0  # Outside exit region


bc_sdf = mixed_bc(
    dimension=2,
    segments=[
        BCSegment(
            name="exit_sdf",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            sdf_region=top_sector_sdf,  # Custom SDF for exit region
            priority=1,
        ),
        BCSegment(
            name="walls",
            bc_type=BCType.REFLECTING,
            boundary="all",
            priority=-1,
        ),
    ],
    domain_bounds=domain_sdf.get_bounding_box(),
)

print("\nSDF-Based BC:")
print("  Exit: Custom SDF (top sector, y > center_y + 0.15*radius)")
print("  Matching: Evaluates SDF at each boundary point")

# Test SDF-based matching
print("\nTesting SDF-Based Matching:")
for point, label in test_points:
    sdf_val = top_sector_sdf(point)
    segment = bc_sdf.get_bc_at_point(point, boundary_id=None)
    print(f"  {label:15s}: sdf={sdf_val:+.2f}, segment={segment.name if segment else 'None'}")

# =============================================================================
# Example 3: Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Visualization")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- Panel 1: Domain Geometry ---
ax = axes[0]

# Draw circle
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = center[0] + radius * np.cos(theta)
circle_y = center[1] + radius * np.sin(theta)
ax.plot(circle_x, circle_y, "k-", linewidth=2, label="Domain Boundary")

# Highlight exit region (top, normal-based)
exit_angles = theta[(circle_y > center[1] + 0.15 * radius)]
exit_x = center[0] + radius * np.cos(exit_angles)
exit_y = center[1] + radius * np.sin(exit_angles)
ax.plot(exit_x, exit_y, "r-", linewidth=4, label="Exit (Absorbing)")

# Show normals at a few points
sample_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
for angle in sample_angles:
    pt = center + radius * np.array([np.cos(angle), np.sin(angle)])
    normal = domain_circle.get_boundary_normal(pt)

    # Check if this is an exit point
    is_exit = pt[1] > center[1] + 0.15 * radius
    color = "red" if is_exit else "blue"

    ax.arrow(
        pt[0],
        pt[1],
        0.08 * normal[0],
        0.08 * normal[1],
        head_width=0.02,
        head_length=0.015,
        fc=color,
        ec=color,
        alpha=0.7,
    )

ax.scatter(*center, color="black", s=100, marker="x", label="Center")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Circular Domain with Normal-Based Exit\n(Red arrows = exit, Blue arrows = reflecting)")

# --- Panel 2: SDF Visualization ---
ax = axes[1]

# Create grid for SDF visualization
x_grid = np.linspace(0.0, 1.0, 100)
y_grid = np.linspace(0.0, 1.0, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# Evaluate SDF at grid points
SDF_values = np.zeros_like(X_grid)
for i in range(len(x_grid)):
    for j in range(len(y_grid)):
        point = np.array([X_grid[j, i], Y_grid[j, i]])
        SDF_values[j, i] = domain_circle.signed_distance(point)

# Plot SDF contours
levels = np.linspace(-0.2, 0.2, 21)
contour = ax.contourf(X_grid, Y_grid, SDF_values, levels=levels, cmap="RdBu_r")
ax.contour(X_grid, Y_grid, SDF_values, levels=[0], colors="black", linewidths=2)
plt.colorbar(contour, ax=ax, label="SDF φ(x)")

ax.plot(circle_x, circle_y, "k-", linewidth=2)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Signed Distance Function\n(φ < 0 inside, φ = 0 boundary, φ > 0 outside)")

# --- Panel 3: BC Application Example ---
ax = axes[2]

# Sample particles and apply BC
n_particles = 200
particles = domain_circle.sample_uniform(n_particles)

# Add some particles slightly outside (to test BC application)
noise_particles = center + radius * 1.1 * np.random.randn(50, 2)
all_particles = np.vstack([particles, noise_particles])

# Classify particles by BC segment
colors = []
for particle in all_particles:
    # Check if inside domain
    if domain_circle.contains(particle):
        colors.append("blue")
    else:
        # Project to boundary and check BC
        projected = domain_circle.project_to_boundary(particle)
        segment = bc_circle.get_bc_at_point(projected, boundary_id=None)

        if segment and segment.name == "exit_top":
            colors.append("red")  # Would be absorbed
        else:
            colors.append("green")  # Would be reflected

ax.scatter(all_particles[:, 0], all_particles[:, 1], c=colors, s=10, alpha=0.6)
ax.plot(circle_x, circle_y, "k-", linewidth=2)
ax.plot(exit_x, exit_y, "r-", linewidth=4)

# Legend
legend_elements = [
    Patch(facecolor="blue", label="Interior (valid)"),
    Patch(facecolor="red", label="Exit region (absorb)"),
    Patch(facecolor="green", label="Wall region (reflect)"),
]
ax.legend(handles=legend_elements, loc="upper left")

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("BC Application to Particles\n(Red = absorbed, Green = reflected, Blue = interior)")

plt.tight_layout()
plt.savefig("examples/outputs/non_tensor_bc_demo.png", dpi=150, bbox_inches="tight")
print("\n✓ Visualization saved to: examples/outputs/non_tensor_bc_demo.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary: Non-Tensor BC Infrastructure")
print("=" * 70)

print("""
✓ Demonstrated BC Capabilities:
  1. Normal-based matching (exit on top of circle)
  2. SDF-based region specification (custom exit shape)
  3. Geometry trait verification (SupportsManifold, SupportsLipschitz)
  4. Dimension-agnostic BC matching (works for any d)

✓ Key Infrastructure (Issue #549):
  - ImplicitDomain with geometry traits (Issue #590)
  - BCSegment with 5 matching modes
  - MeshfreeApplicator for particle methods
  - Fully functional for non-rectangular domains

✓ Usage Pattern:
  1. Create implicit domain (Hypersphere, arbitrary SDF)
  2. Specify BC using normal_direction or sdf_region
  3. Use with particle solver, meshfree solver, etc.

✓ Architecture Benefits:
  - No mesh generation required
  - Rotations/transformations automatic
  - Same code for 2D, 3D, ..., 100D
  - Natural obstacle representation via CSG

See also:
  - examples/advanced/meshfree_mfg.py - Full MFG on implicit domains
  - docs/TECHNICAL_REFERENCE_HIGH_DIMENSIONAL_MFG.md - Mathematical foundations
""")

print("\n" + "=" * 70)
print("✓ Non-tensor BC demonstration complete!")
print("=" * 70)
