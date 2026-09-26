#!/usr/bin/env python3
"""
Dual Geometry Example: FEM Mesh + Regular Grid
==============================================

Demonstrates using FEM unstructured mesh for one solver and regular grid for another.

**Use Case**: Complex domain with obstacles (FEM mesh) + regular grid for efficient HJB.

**Current Status**: Full support with automatic Delaunay interpolation (scipy required).
**Fallback**: Nearest neighbor if scipy not available.

**Mathematical Setup**:
- HJB: Regular grid (fast value iteration)
- FP: FEM mesh (handles complex boundaries naturally)
- Projection: Grid↔Mesh using nearest neighbor (fallback) or custom Delaunay
"""

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.geometry import Mesh2D, ProjectionRegistry, TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def demonstrate_fem_mesh_projection_basic():
    """Demonstrate FEM mesh projection with automatic method selection."""

    print("=" * 70)
    print("FEM Mesh Projection - Automatic Method Selection")
    print("=" * 70)

    # Create FEM mesh for complex domain (e.g., rectangle with hole)
    print("\nCreating FEM mesh (rectangle with circular hole)...")

    mesh = Mesh2D(
        domain_type="rectangle",
        bounds=(0.0, 1.0, 0.0, 1.0),
        holes=[{"type": "circle", "center": (0.5, 0.5), "radius": 0.2}],
        mesh_size=0.05,  # Element size
    )

    # Generate mesh using Gmsh
    try:
        mesh_data = mesh.generate_mesh()
        print(f"✓ Mesh generated: {mesh_data.num_vertices} vertices, {mesh_data.num_elements} elements")
    except ImportError:
        print("[WARN]  Gmsh not available - using demonstration with TensorProductGrid instead")
        print("   Install gmsh: pip install gmsh")
        # Fallback to grid for demonstration
        mesh = TensorProductGrid(
            bounds=[(0, 1), (0, 1)], Nx_points=[21, 21], boundary_conditions=no_flux_bc(dimension=2)
        )
        mesh_data = None

    # Create regular grid for HJB
    grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx_points=[51, 51], boundary_conditions=no_flux_bc(dimension=2))

    grid_shape = grid.get_grid_shape()
    print(f"\nGrid points: {grid_shape[0] * grid_shape[1]:,}")
    if mesh_data:
        print(f"Mesh vertices: {mesh_data.num_vertices:,}")

    # Create dual geometry problem
    print("\nCreating MFG problem with dual geometries...")
    problem = MFGProblem(
        hjb_geometry=grid,  # Regular grid for HJB
        fp_geometry=mesh,  # FEM mesh for FP
        time_domain=(1.0, 50),
        volatility=0.1,
    )

    # Check projection methods
    projector = problem.geometry_projector
    print("\nProjection methods selected:")
    print(f"  Grid→Mesh (HJB→FP): {projector.hjb_to_fp_method}")
    print(f"  Mesh→Grid (FP→HJB): {projector.fp_to_hjb_method}")

    if projector.fp_to_hjb_method == "registry":
        print("\n✓ Using optimized Delaunay interpolation (automatic)")
    else:
        print("\n[WARN]  Using nearest neighbor fallback (scipy not available)")
        print("   Install scipy for better accuracy: pip install scipy")

    # Demonstrate projection
    print("\nTesting projections...")

    # Create test data on grid
    U_grid = np.random.rand(*grid.get_grid_shape())

    # Project to mesh
    U_mesh = projector.project_hjb_to_fp(U_grid)
    if mesh_data:
        print(f"  ✓ Grid→Mesh: {U_grid.shape} → {U_mesh.shape} (mesh vertices)")
    else:
        print(f"  ✓ Grid→Mesh: {U_grid.shape} → {U_mesh.shape}")

    # Create test data on mesh
    mesh_points = mesh.get_spatial_grid()
    M_mesh = np.random.rand(len(mesh_points))

    # Project to grid
    M_grid = projector.project_fp_to_hjb(M_mesh)
    print(f"  ✓ Mesh→Grid: {M_mesh.shape} → {M_grid.shape}")

    print("\n✓ Basic FEM mesh projection working (via nearest neighbor)")


def register_specialized_fem_projections():
    """
    Demonstrate manual registration of FEM mesh projections.

    Note: FEM mesh projections are now registered AUTOMATICALLY on module import
    if scipy is available. This function demonstrates how to add custom projections.
    """

    print("\n" + "=" * 70)
    print("FEM Mesh Projections - Already Registered Automatically!")
    print("=" * 70)

    print("\n✓ FEM mesh projections are now part of the core system")
    print("  Registered automatically when scipy is available")
    print("  No manual registration needed for basic FEM mesh support")

    # Register Mesh → Grid projection with Delaunay interpolation
    @ProjectionRegistry.register(Mesh2D, TensorProductGrid, "fp_to_hjb")
    def mesh_to_grid_delaunay(mesh_geo, grid_geo, mesh_values, **kwargs):
        """
        Project from FEM mesh to regular grid using Delaunay interpolation.

        This provides better accuracy than nearest neighbor by using the
        mesh's triangulation for linear interpolation.

        Args:
            mesh_geo: Source FEM mesh
            grid_geo: Target regular grid
            mesh_values: Values at mesh vertices (N_vertices,)

        Returns:
            Values on grid (nx+1, ny+1)
        """
        try:
            from scipy.interpolate import LinearNDInterpolator
        except ImportError:
            raise ImportError("scipy required for Delaunay interpolation") from None

        # Get mesh vertex positions
        vertices = mesh_geo.get_spatial_grid()  # (N_vertices, 2)

        # Create Delaunay interpolator (automatically uses mesh triangulation)
        interpolator = LinearNDInterpolator(vertices, mesh_values)

        # Evaluate at grid points
        grid_points = grid_geo.get_spatial_grid()  # (N_grid, 2)
        grid_values_flat = interpolator(grid_points)

        # Handle extrapolation (points outside mesh convex hull → NaN)
        # Fill with nearest neighbor
        nan_mask = np.isnan(grid_values_flat)
        if np.any(nan_mask):
            from scipy.spatial import KDTree

            tree = KDTree(vertices)
            _, nearest_indices = tree.query(grid_points[nan_mask])
            grid_values_flat[nan_mask] = mesh_values[nearest_indices]

        # Reshape to grid
        grid_shape = grid_geo.get_grid_shape()
        return grid_values_flat.reshape(grid_shape)

    # Register Grid → Mesh projection (already optimal with interpolation)
    @ProjectionRegistry.register(TensorProductGrid, Mesh2D, "hjb_to_fp")
    def grid_to_mesh_interpolation(grid_geo, mesh_geo, grid_values, **kwargs):
        """
        Project from regular grid to FEM mesh using bilinear interpolation.

        This is already optimal - grid's built-in interpolator is efficient.

        Args:
            grid_geo: Source regular grid
            mesh_geo: Target FEM mesh
            grid_values: Values on grid (nx+1, ny+1)

        Returns:
            Values at mesh vertices (N_vertices,)
        """
        # Get mesh vertex positions
        vertices = mesh_geo.get_spatial_grid()  # (N_vertices, 2)

        # Use grid's built-in interpolator (bilinear)
        interpolator = grid_geo.get_interpolator()
        mesh_values = interpolator(grid_values, vertices)

        return mesh_values

    print("✓ Registered specialized projections:")
    print("  - Mesh2D → TensorProductGrid: Delaunay interpolation")
    print("  - TensorProductGrid → Mesh2D: Bilinear interpolation")

    # Verify registration
    registered = ProjectionRegistry.list_registered()
    fem_projections = [key for key in registered if "Mesh2D" in str(key)]
    print(f"\n✓ Found {len(fem_projections)} FEM mesh projections in registry")


def demonstrate_fem_mesh_projection_optimized():
    """Demonstrate optimized FEM mesh projection with specialized methods."""

    print("\n" + "=" * 70)
    print("FEM Mesh Projection - Optimized (Delaunay Interpolation)")
    print("=" * 70)

    # Create geometries
    mesh = Mesh2D(
        domain_type="rectangle",
        bounds=(0.0, 1.0, 0.0, 1.0),
        holes=[{"type": "circle", "center": (0.5, 0.5), "radius": 0.2}],
        mesh_size=0.05,
    )

    grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx_points=[51, 51], boundary_conditions=no_flux_bc(dimension=2))

    try:
        mesh_data = mesh.generate_mesh()
        print(f"✓ Mesh generated: {mesh_data.num_vertices} vertices")
    except ImportError:
        print("[WARN]  Gmsh not available - skipping optimized demonstration")
        return

    # Create problem (will now use registered specialized projections)
    print("\nCreating MFG problem with optimized projections...")
    problem = MFGProblem(hjb_geometry=grid, fp_geometry=mesh, time_domain=(1.0, 50), diffusion=0.1)

    projector = problem.geometry_projector
    print("\nProjection methods selected:")
    print(f"  Grid→Mesh: {projector.hjb_to_fp_method}")
    print(f"  Mesh→Grid: {projector.fp_to_hjb_method}")

    if projector.fp_to_hjb_method == "registry":
        print("\n✓ Using optimized Delaunay interpolation (registered)")
    else:
        print("\n[WARN]  Still using fallback (register projections first)")

    # Test accuracy improvement
    print("\nTesting projection accuracy...")

    # Create smooth test function
    mesh_vertices = mesh.get_spatial_grid()

    def test_func(x, y):
        return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

    M_mesh_exact = test_func(mesh_vertices[:, 0], mesh_vertices[:, 1])

    # Project to grid
    M_grid = projector.project_fp_to_hjb(M_mesh_exact)

    # Check error by projecting back
    grid_points = grid.get_spatial_grid()
    M_grid_exact = test_func(grid_points[:, 0], grid_points[:, 1])
    error = np.abs(M_grid.ravel() - M_grid_exact)

    print(f"  Projection error: max={error.max():.4e}, mean={error.mean():.4e}")
    print("  (Lower error indicates better interpolation quality)")


def compare_projection_methods():
    """Compare nearest neighbor vs Delaunay interpolation accuracy."""

    print("\n" + "=" * 70)
    print("Comparison: Nearest Neighbor vs Delaunay Interpolation")
    print("=" * 70)

    print("\nProjection Method Characteristics:")
    print("\n1. Nearest Neighbor (Default Fallback):")
    print("   ✓ Always works (no extra dependencies)")
    print("   ✓ Fast (O(log N) with KD-tree)")
    print("   ✗ Piecewise constant (discontinuous)")
    print("   ✗ Lower accuracy for smooth functions")
    print("   Use case: Quick prototyping, coarse estimates")

    print("\n2. Delaunay Interpolation (Optimal for FEM):")
    print("   ✓ Respects mesh triangulation")
    print("   ✓ Linear interpolation (C0 continuous)")
    print("   ✓ Higher accuracy for smooth functions")
    print("   ✗ Requires scipy")
    print("   ✗ Slightly slower (O(N log N) setup)")
    print("   Use case: Production accuracy, mesh-based methods")

    print("\n3. When to Use Each:")
    print("   - Nearest Neighbor: Rapid development, debugging, coarse problems")
    print("   - Delaunay: Final production runs, accuracy-critical applications")
    print("   - Custom: Specialized needs (conservative, high-order)")


if __name__ == "__main__":
    import os

    if "MPLBACKEND" not in os.environ:
        import matplotlib

        matplotlib.use("Agg")

    try:
        # Demonstrate basic support (nearest neighbor fallback)
        demonstrate_fem_mesh_projection_basic()

        # Register specialized projections
        register_specialized_fem_projections()

        # Demonstrate optimized support (Delaunay interpolation)
        demonstrate_fem_mesh_projection_optimized()

        # Comparison
        compare_projection_methods()

        print("\n" + "=" * 70)
        print("Summary: FEM Mesh Projection")
        print("=" * 70)
        print("\n✓ Delaunay interpolation registered AUTOMATICALLY (scipy)")
        print("✓ Nearest neighbor fallback if scipy not available")
        print("✓ Production-ready for mesh-based MFG problems")
        print("\nKey Features:")
        print("  1. Automatic method selection based on available libraries")
        print("  2. Delaunay interpolation preserves mesh triangulation")
        print("  3. Graceful degradation to nearest neighbor without scipy")
        print("  4. Easy to add custom projections via registry pattern")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        raise
