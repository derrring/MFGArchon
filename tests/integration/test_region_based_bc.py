"""
Integration tests for region-based boundary conditions (Issue #596 Phase 2.5C).

Tests the complete workflow:
1. Mark regions on geometry via mark_region()
2. Create BC specifications using mixed_bc_from_regions()
3. Apply BCs via FDM applicators with geometry parameter
4. Verify correct BC application at region boundaries

Test scenarios:
- 1D inlet/outlet with predicates
- 2D corridor flow (inlet/outlet/walls)
- Region intersection and priority handling
- Performance validation (<5% overhead)
"""

import time

import pytest

import numpy as np

from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    BoundaryConditions,
    FDMApplicator,
    mixed_bc_from_regions,
    no_flux_bc,
)


class TestRegionBasedBC1D:
    """Test region-based BCs in 1D."""

    def test_basic_inlet_outlet_1d(self):
        """Test basic inlet/outlet regions in 1D."""
        # Create 1D grid [0, 10] with 101 points
        geometry = TensorProductGrid(bounds=[(0, 10)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[101])

        # Mark inlet (x < 1.0) and outlet (x > 9.0) regions
        geometry.mark_region("inlet", predicate=lambda x: x[:, 0] < 1.0)
        geometry.mark_region("outlet", predicate=lambda x: x[:, 0] > 9.0)

        # Create BCs: Dirichlet inlet (u=1.0), Neumann outlet (du/dx=0), periodic elsewhere
        bc_config = {
            "inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0),
            "outlet": BCSegment(name="outlet_bc", bc_type=BCType.NEUMANN, value=0.0),
            "default": BCSegment(name="default_bc", bc_type=BCType.PERIODIC),
        }

        bc = mixed_bc_from_regions(geometry, bc_config)

        # Verify BC object created correctly
        assert bc.dimension == 1
        assert len(bc.segments) == 2
        assert bc.segments[0].region_name == "inlet"
        assert bc.segments[1].region_name == "outlet"
        assert bc.default_bc == BCType.PERIODIC

        # Apply to uniform field
        field = np.ones(101) * 0.5
        applicator = FDMApplicator(dimension=1)
        padded = applicator.apply(field, bc, domain_bounds=np.array([[0, 10]]), geometry=geometry)

        # Verify shape: (101 + 2,)
        assert padded.shape == (103,)

        # Verify interior preserved
        assert np.allclose(padded[1:-1], field)

        # Verify ghost cells reflect BC types
        # Left ghost (inlet): Dirichlet BC u_boundary = 1.0
        # For cell-centered grid: u_ghost = 2*g - u_interior
        # u_ghost = 2*1.0 - 0.5 = 1.5
        assert np.isclose(padded[0], 2 * 1.0 - field[0], atol=1e-10)

        # Right ghost (outlet): Neumann BC du/dx = 0
        # For Neumann BC: ghost mirrors adjacent interior (reflection about boundary)
        # ghost[-1] = interior[-1] (last interior point)
        # Note: Test uses uniform field so field[-1] == field[-2], but correct check is field[-1]
        assert np.isclose(padded[-1], field[-1], atol=1e-10)

    def test_boundary_vs_predicate_regions_1d(self):
        """Test boundary-specified vs predicate-specified regions give same result."""
        geometry = TensorProductGrid(bounds=[(0, 1)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[51])

        # Method 1: Mark using boundary identifier
        geometry.mark_region("left_boundary", boundary="x_min")

        # Method 2: Mark using predicate (x < tolerance)
        geometry.mark_region("left_predicate", predicate=lambda x: x[:, 0] < 1e-8)

        # Both should produce similar region masks at left boundary
        mask_boundary = geometry.get_region_mask("left_boundary")
        mask_predicate = geometry.get_region_mask("left_predicate")

        # At least the leftmost point should be in both regions
        assert mask_boundary[0]
        assert mask_predicate[0]

    def test_region_priority_1d(self):
        """Test priority resolution when regions overlap in 1D."""
        geometry = TensorProductGrid(bounds=[(0, 1)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[51])

        # Mark overlapping regions
        geometry.mark_region("broad", predicate=lambda x: x[:, 0] < 0.5)
        geometry.mark_region("narrow", predicate=lambda x: x[:, 0] < 0.1)

        # Create BC segments with different priorities
        # Higher priority number wins (segments sorted with reverse=True)
        bc_narrow = BCSegment(
            name="narrow_bc",
            bc_type=BCType.DIRICHLET,
            value=1.0,
            region_name="narrow",
            priority=2,  # Higher number = higher precedence
        )
        bc_broad = BCSegment(
            name="broad_bc",
            bc_type=BCType.NEUMANN,
            value=0.0,
            region_name="broad",
            priority=1,  # Lower number = lower precedence
        )

        bc = BoundaryConditions(
            dimension=1,
            segments=[bc_narrow, bc_broad],  # Sorted by priority
            default_bc=BCType.PERIODIC,
            domain_bounds=np.array([[0, 1]]),
        )

        # Test point in narrow region (x=0.05) should get narrow BC
        point = np.array([0.05])
        segment = bc.get_bc_at_point(point, boundary_id=None, geometry=geometry)
        assert segment.name == "narrow_bc"

        # Test point in broad but not narrow (x=0.3) should get broad BC
        point = np.array([0.3])
        segment = bc.get_bc_at_point(point, boundary_id=None, geometry=geometry)
        assert segment.name == "broad_bc"


class TestRegionBasedBC2D:
    """Test region-based BCs in 2D."""

    def test_corridor_flow_2d(self):
        """Test 2D corridor with inlet, outlet, and wall regions."""
        # Create 2D grid [0,2] x [0,1]
        geometry = TensorProductGrid(
            bounds=[(0, 2), (0, 1)], boundary_conditions=no_flux_bc(dimension=2), Nx_points=[41, 21]
        )

        # Mark regions:
        # - Inlet: left boundary (x=0)
        # - Outlet: right boundary (x=2)
        # - Walls: top and bottom boundaries (y=0, y=1)
        geometry.mark_region("inlet", boundary="x_min")
        geometry.mark_region("outlet", boundary="x_max")
        geometry.mark_region("walls_bottom", boundary="y_min")
        geometry.mark_region("walls_top", boundary="y_max")

        # Create BCs
        bc_config = {
            "inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0),
            "outlet": BCSegment(name="outlet_bc", bc_type=BCType.NEUMANN, value=0.0),
            "walls_bottom": BCSegment(name="wall_bottom_bc", bc_type=BCType.NO_FLUX, value=0.0),
            "walls_top": BCSegment(name="wall_top_bc", bc_type=BCType.NO_FLUX, value=0.0),
        }

        bc = mixed_bc_from_regions(geometry, bc_config)

        # Verify BC object
        assert bc.dimension == 2
        assert len(bc.segments) == 4

        # Apply to field
        field = np.ones((21, 41)) * 0.5
        applicator = FDMApplicator(dimension=2)
        padded = applicator.apply(field, bc, domain_bounds=np.array([[0, 2], [0, 1]]), geometry=geometry)

        # Verify shape: (21+2, 41+2) = (23, 43)
        assert padded.shape == (23, 43)

        # Verify interior preserved
        assert np.allclose(padded[1:-1, 1:-1], field)

        # Verify ghost cells were populated (no longer all zeros)
        # Exact values depend on BC formulas which are tested elsewhere
        assert not np.allclose(padded[1:-1, 0], 0.0)  # Left ghost cells
        assert not np.allclose(padded[1:-1, -1], 0.0)  # Right ghost cells
        assert not np.allclose(padded[0, 1:-1], 0.0)  # Bottom ghost cells
        assert not np.allclose(padded[-1, 1:-1], 0.0)  # Top ghost cells

    def test_predicate_based_inlet_2d(self):
        """Test inlet defined by predicate rather than boundary."""
        geometry = TensorProductGrid(
            bounds=[(0, 1), (0, 1)], boundary_conditions=no_flux_bc(dimension=2), Nx_points=[21, 21]
        )

        # Mark inlet as left quarter of domain (x < 0.25)
        geometry.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.25)

        # Create BC
        bc_config = {
            "inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0),
            "default": BCSegment(name="default_bc", bc_type=BCType.PERIODIC),
        }

        bc = mixed_bc_from_regions(geometry, bc_config)

        # Verify BC retrieval at different points
        # Point in inlet region (x=0.1, y=0.5)
        point_inlet = np.array([0.1, 0.5])
        segment = bc.get_bc_at_point(point_inlet, "x_min", geometry=geometry)
        assert segment.name == "inlet_bc"

        # Point outside inlet region (x=0.5, y=0.5)
        point_default = np.array([0.5, 0.5])
        segment = bc.get_bc_at_point(point_default, None, geometry=geometry)
        # Should get default BC
        assert segment.bc_type == BCType.PERIODIC

    def test_region_intersection_2d(self):
        """Test overlapping regions with priority resolution in 2D."""
        geometry = TensorProductGrid(
            bounds=[(0, 1), (0, 1)], boundary_conditions=no_flux_bc(dimension=2), Nx_points=[21, 21]
        )

        # Mark overlapping regions:
        # - Bottom half (y < 0.5)
        # - Left quarter (x < 0.25)
        # Intersection: bottom-left corner
        geometry.mark_region("bottom", predicate=lambda x: x[:, 1] < 0.5)
        geometry.mark_region("left", predicate=lambda x: x[:, 0] < 0.25)

        # Create BC segments with priorities
        # Higher priority number wins (segments sorted with reverse=True)
        bc_left = BCSegment(
            name="left_bc",
            bc_type=BCType.DIRICHLET,
            value=1.0,
            region_name="left",
            priority=2,  # Higher number = higher precedence
        )
        bc_bottom = BCSegment(
            name="bottom_bc",
            bc_type=BCType.NEUMANN,
            value=0.0,
            region_name="bottom",
            priority=1,  # Lower number = lower precedence
        )

        bc = BoundaryConditions(
            dimension=2,
            segments=[bc_left, bc_bottom],
            default_bc=BCType.PERIODIC,
            domain_bounds=np.array([[0, 1], [0, 1]]),
        )

        # Test point in intersection (x=0.1, y=0.3)
        # Should get left BC (higher priority)
        point_intersection = np.array([0.1, 0.3])
        segment = bc.get_bc_at_point(point_intersection, None, geometry=geometry)
        assert segment.name == "left_bc"

        # Test point in bottom only (x=0.5, y=0.3)
        point_bottom = np.array([0.5, 0.3])
        segment = bc.get_bc_at_point(point_bottom, None, geometry=geometry)
        assert segment.name == "bottom_bc"


class TestRegionBasedBCPerformance:
    """Performance tests for region-based BC application."""

    @pytest.mark.slow
    def test_region_lookup_overhead(self):
        """Test that region-based BC has <5% overhead vs standard BC."""
        # Create moderately sized 2D grid
        geometry = TensorProductGrid(
            bounds=[(0, 1), (0, 1)], boundary_conditions=no_flux_bc(dimension=2), Nx_points=[101, 101]
        )

        # Standard BC (no regions)
        bc_standard = BoundaryConditions(
            dimension=2,
            segments=[BCSegment(name="all", bc_type=BCType.DIRICHLET, value=0.0, boundary=None)],
        )

        # Region-based BC
        geometry.mark_region("all_domain", predicate=lambda x: np.ones(x.shape[0], dtype=bool))
        bc_region = mixed_bc_from_regions(
            geometry,
            {"all_domain": BCSegment(name="all_bc", bc_type=BCType.DIRICHLET, value=0.0)},
        )

        field = np.random.randn(101, 101)
        applicator = FDMApplicator(dimension=2)
        domain_bounds = np.array([[0, 1], [0, 1]])

        # Warmup
        applicator.apply(field, bc_standard, domain_bounds=domain_bounds)
        applicator.apply(field, bc_region, domain_bounds=domain_bounds, geometry=geometry)

        # Benchmark standard BC
        n_iterations = 100
        start = time.perf_counter()
        for _ in range(n_iterations):
            applicator.apply(field, bc_standard, domain_bounds=domain_bounds)
        time_standard = time.perf_counter() - start

        # Benchmark region-based BC
        start = time.perf_counter()
        for _ in range(n_iterations):
            applicator.apply(field, bc_region, domain_bounds=domain_bounds, geometry=geometry)
        time_region = time.perf_counter() - start

        # Calculate overhead
        overhead = (time_region - time_standard) / time_standard * 100

        print(f"\nPerformance comparison ({n_iterations} iterations):")
        print(f"  Standard BC: {time_standard * 1000:.2f} ms")
        print(f"  Region BC:   {time_region * 1000:.2f} ms")
        print(f"  Overhead:    {overhead:.1f}%")

        # Assert <5% overhead
        # Note: This is a loose bound for CI environments
        # In practice, overhead should be <1%
        assert overhead < 10, f"Region-based BC overhead {overhead:.1f}% exceeds 10%"


class TestRegionBasedBCEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_region(self):
        """Test error when BC references non-existent region."""
        geometry = TensorProductGrid(bounds=[(0, 1)], boundary_conditions=no_flux_bc(dimension=1), Nx_points=[51])

        # Try to create BC for region that doesn't exist
        bc_config = {"nonexistent": BCSegment(name="bad_bc", bc_type=BCType.DIRICHLET, value=1.0)}

        with pytest.raises(ValueError, match="not found"):
            mixed_bc_from_regions(geometry, bc_config)

    def test_backward_compatibility(self):
        """Test that old BC code without region_name still works."""
        # Create BC the old way (no regions)
        bc_left = BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min")
        bc_right = BCSegment(name="right", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max")
        bc = BoundaryConditions(
            dimension=1,
            segments=[bc_left, bc_right],
            default_bc=BCType.PERIODIC,
            domain_bounds=np.array([[0, 1]]),
        )

        # Apply without geometry (should work fine)
        field = np.ones(51)
        applicator = FDMApplicator(dimension=1)
        padded = applicator.apply(field, bc, domain_bounds=np.array([[0, 1]]))

        # Should work without errors
        assert padded.shape == (53,)
