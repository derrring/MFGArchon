"""
Unit tests for dimension-agnostic mixed boundary conditions.

Tests BCSegment, BCType, and MixedBoundaryConditions classes.
"""

from __future__ import annotations

import numpy as np

from mfgarchon.geometry import BCSegment, BCType, MixedBoundaryConditions
from mfgarchon.geometry.boundary import no_flux_bc


class TestBCType:
    """Test BCType enum."""

    def test_bc_types_defined(self):
        """Test that all BC types are defined."""
        assert BCType.DIRICHLET.value == "dirichlet"
        assert BCType.NEUMANN.value == "neumann"
        assert BCType.ROBIN.value == "robin"
        assert BCType.PERIODIC.value == "periodic"
        assert BCType.REFLECTING.value == "reflecting"
        assert BCType.NO_FLUX.value == "no_flux"


class TestBCSegment:
    """Test BCSegment class."""

    def test_basic_creation(self):
        """Test creating a basic BC segment."""
        segment = BCSegment(name="test", bc_type=BCType.DIRICHLET, value=1.0)
        assert segment.name == "test"
        assert segment.bc_type == BCType.DIRICHLET
        assert segment.value == 1.0
        assert segment.boundary is None
        assert segment.region is None
        assert segment.priority == 0

    def test_1d_segment(self):
        """Test 1D boundary segment (e.g., left wall)."""
        segment = BCSegment(
            name="left_wall",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="left",
        )

        domain_bounds = np.array([[0.0, 10.0]])  # 1D domain [0, 10]

        # Point on left boundary matches
        assert segment.matches_point(
            point=np.array([0.0]),
            boundary_id="left",
            domain_bounds=domain_bounds,
        )

        # Point on right boundary doesn't match
        assert not segment.matches_point(
            point=np.array([10.0]),
            boundary_id="right",
            domain_bounds=domain_bounds,
        )

    def test_2d_segment_with_region(self):
        """Test 2D segment with region constraint (exit on right wall, y in [4.25, 5.75])."""
        exit_segment = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="right",
            region={"y": (4.25, 5.75)},
        )

        domain_bounds = np.array([[0.0, 10.0], [0.0, 10.0]])  # 2D domain [0,10]^2

        # Point on right wall within y-range matches
        assert exit_segment.matches_point(
            point=np.array([10.0, 5.0]),
            boundary_id="right",
            domain_bounds=domain_bounds,
        )

        # Point on right wall outside y-range doesn't match
        assert not exit_segment.matches_point(
            point=np.array([10.0, 2.0]),
            boundary_id="right",
            domain_bounds=domain_bounds,
        )

        # Point on top wall (even within y-range) doesn't match
        assert not exit_segment.matches_point(
            point=np.array([5.0, 10.0]),
            boundary_id="top",
            domain_bounds=domain_bounds,
        )

    def test_2d_segment_all_boundaries(self):
        """Test segment that applies to all boundaries."""
        wall_segment = BCSegment(
            name="walls",
            bc_type=BCType.NEUMANN,
            value=0.0,
            boundary="all",
        )

        domain_bounds = np.array([[0.0, 10.0], [0.0, 10.0]])

        # Matches any boundary
        assert wall_segment.matches_point(
            point=np.array([0.0, 5.0]),
            boundary_id="left",
            domain_bounds=domain_bounds,
        )
        assert wall_segment.matches_point(
            point=np.array([10.0, 5.0]),
            boundary_id="right",
            domain_bounds=domain_bounds,
        )
        assert wall_segment.matches_point(
            point=np.array([5.0, 0.0]),
            boundary_id="bottom",
            domain_bounds=domain_bounds,
        )
        assert wall_segment.matches_point(
            point=np.array([5.0, 10.0]),
            boundary_id="top",
            domain_bounds=domain_bounds,
        )

    def test_segment_with_int_boundary(self):
        """Test segment with integer Gmsh physical group tag (#732 Tier 1)."""
        segment = BCSegment(
            name="gmsh_wall",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary=42,
        )

        domain_bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        point = np.array([0.5, 0.5])

        # Matches integer boundary_id
        assert segment.matches_point(point, boundary_id=42, domain_bounds=domain_bounds)
        # Does not match different int
        assert not segment.matches_point(point, boundary_id=99, domain_bounds=domain_bounds)
        # Does not match string
        assert not segment.matches_point(point, boundary_id="left", domain_bounds=domain_bounds)
        # Does not match None
        assert not segment.matches_point(point, boundary_id=None, domain_bounds=domain_bounds)

    def test_3d_segment_with_multiple_constraints(self):
        """Test 3D segment with constraints on multiple axes."""
        segment = BCSegment(
            name="corner",
            bc_type=BCType.ROBIN,
            value=1.0,
            region={"x": (0, 1), "y": (0, 1), "z": (9, 10)},
        )

        domain_bounds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])  # 3D domain [0,10]^3

        # Point within all constraints matches
        assert segment.matches_point(
            point=np.array([0.5, 0.5, 9.5]),
            boundary_id="top",
            domain_bounds=domain_bounds,
        )

        # Point violating one constraint doesn't match
        assert not segment.matches_point(
            point=np.array([0.5, 2.0, 9.5]),  # y = 2.0 violates y ∈ [0, 1]
            boundary_id="top",
            domain_bounds=domain_bounds,
        )

    def test_get_value_constant(self):
        """Test getting constant BC value."""
        segment = BCSegment(name="test", bc_type=BCType.DIRICHLET, value=5.0)
        assert segment.get_value(np.array([1.0]), t=0.0) == 5.0
        assert segment.get_value(np.array([1.0, 2.0]), t=1.0) == 5.0

    def test_get_value_callable_1d(self):
        """Test getting callable BC value in 1D."""
        segment = BCSegment(
            name="test",
            bc_type=BCType.DIRICHLET,
            value=lambda x, t: x**2 + t,
        )
        assert segment.get_value(np.array([2.0]), t=1.0) == 5.0

    def test_get_value_callable_2d(self):
        """Test getting callable BC value in 2D."""
        segment = BCSegment(
            name="test",
            bc_type=BCType.DIRICHLET,
            value=lambda x, y, t: x + 2 * y + t,
        )
        assert segment.get_value(np.array([1.0, 2.0]), t=3.0) == 8.0

    def test_string_representation(self):
        """Test string representation."""
        segment = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="right",
            region={"y": (4.25, 5.75)},
            priority=1,
        )
        s = str(segment)
        assert "exit" in s
        assert "dirichlet" in s
        assert "right" in s
        assert "y" in s
        assert "priority=1" in s

    # =========================================================================
    # Region-Based BC Tests (Issue #596 Phase 2.5)
    # =========================================================================

    def test_region_name_field(self):
        """Test BCSegment with region_name field (Issue #596 Phase 2.5)."""
        segment = BCSegment(
            name="inlet_bc",
            bc_type=BCType.DIRICHLET,
            value=1.0,
            region_name="inlet",
        )

        assert segment.name == "inlet_bc"
        assert segment.bc_type == BCType.DIRICHLET
        assert segment.value == 1.0
        assert segment.region_name == "inlet"
        assert segment.boundary is None
        assert segment.region is None
        assert segment.sdf_region is None
        assert segment.normal_direction is None

    def test_region_name_backward_compatibility(self):
        """Test region_name defaults to None (backward compatibility)."""
        segment = BCSegment(
            name="left",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="x_min",
        )

        assert segment.region_name is None
        assert segment.boundary == "x_min"

    def test_region_name_validation_rejects_multiple_specs(self):
        """Test validation rejects multiple region specifications."""
        import pytest

        # Conflict: boundary + region_name (different categories)
        with pytest.raises(ValueError, match="Cannot mix region specification methods"):
            BCSegment(
                name="bad",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                boundary="x_min",
                region_name="inlet",
            )

        # Conflict: region + region_name (different categories)
        with pytest.raises(ValueError, match="Cannot mix region specification methods"):
            BCSegment(
                name="bad",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                region={"y": (0.4, 0.6)},
                region_name="inlet",
            )

        # Conflict: sdf_region + region_name (different categories)
        with pytest.raises(ValueError, match="Cannot mix region specification methods"):
            BCSegment(
                name="bad",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                sdf_region=lambda x: np.linalg.norm(x) - 1.0,
                region_name="inlet",
            )

        # Conflict: normal_direction + region_name (different categories)
        with pytest.raises(ValueError, match="Cannot mix region specification methods"):
            BCSegment(
                name="bad",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                normal_direction=np.array([0.0, 1.0]),
                region_name="inlet",
            )

    def test_region_name_validation_accepts_single_spec(self):
        """Test validation accepts single region specification."""
        # Each of these should succeed without raising
        seg1 = BCSegment(name="s1", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_min")
        assert seg1.boundary == "x_min"

        seg2 = BCSegment(name="s2", bc_type=BCType.DIRICHLET, value=0.0, region={"y": (0, 1)})
        assert seg2.region == {"y": (0, 1)}

        seg3 = BCSegment(name="s3", bc_type=BCType.DIRICHLET, value=0.0, region_name="inlet")
        assert seg3.region_name == "inlet"

        seg4 = BCSegment(name="s4", bc_type=BCType.DIRICHLET, value=0.0, sdf_region=lambda x: np.linalg.norm(x) - 1.0)
        assert seg4.sdf_region is not None

        seg5 = BCSegment(name="s5", bc_type=BCType.DIRICHLET, value=0.0, normal_direction=np.array([0.0, 1.0]))
        assert seg5.normal_direction is not None

    def test_region_name_validation_allows_no_spec(self):
        """Test validation allows no region specification (uniform BC)."""
        segment = BCSegment(name="uniform", bc_type=BCType.DIRICHLET, value=0.0)

        assert segment.boundary is None
        assert segment.region is None
        assert segment.sdf_region is None
        assert segment.normal_direction is None
        assert segment.region_name is None


class TestMixedBCFromRegions:
    """Test mixed_bc_from_regions() helper function (Issue #596 Phase 2.5)."""

    def test_basic_usage(self):
        """Test basic usage of mixed_bc_from_regions()."""
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import mixed_bc_from_regions

        # Setup geometry with marked regions (placeholder BC, will be replaced by mixed_bc_from_regions)
        geometry = TensorProductGrid(
            bounds=[(0, 1), (0, 1)], Nx_points=[50, 50], boundary_conditions=no_flux_bc(dimension=2)
        )
        geometry.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.1)
        geometry.mark_region("outlet", boundary="x_max")

        # Define BCs via dictionary
        bc_config = {
            "inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0),
            "outlet": BCSegment(name="outlet_bc", bc_type=BCType.NEUMANN, value=0.0),
            "default": BCSegment(name="default_bc", bc_type=BCType.PERIODIC),
        }

        # Create boundary conditions
        bc = mixed_bc_from_regions(geometry, bc_config)

        assert bc.dimension == 2
        assert len(bc.segments) == 2  # inlet + outlet (default not in segments)
        assert bc.segments[0].region_name == "inlet"
        assert bc.segments[1].region_name == "outlet"
        assert bc.default_bc == BCType.PERIODIC
        assert bc.default_value == 0.0

    def test_auto_populates_region_name(self):
        """Test that region_name is auto-populated from config keys."""
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import mixed_bc_from_regions

        geometry = TensorProductGrid(bounds=[(0, 10)], Nx_points=[101], boundary_conditions=no_flux_bc(dimension=1))
        geometry.mark_region("left", boundary="x_min")
        geometry.mark_region("right", boundary="x_max")

        bc_config = {
            "left": BCSegment(name="left_bc", bc_type=BCType.DIRICHLET, value=0.0),
            "right": BCSegment(name="right_bc", bc_type=BCType.DIRICHLET, value=1.0),
        }

        bc = mixed_bc_from_regions(geometry, bc_config)

        # Check region_name was populated
        assert bc.segments[0].region_name == "left"
        assert bc.segments[1].region_name == "right"

        # Check original segment fields preserved
        assert bc.segments[0].name == "left_bc"
        assert bc.segments[0].value == 0.0
        assert bc.segments[1].name == "right_bc"
        assert bc.segments[1].value == 1.0

    def test_infers_dimension_from_geometry(self):
        """Test dimension inference from geometry."""
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import mixed_bc_from_regions

        geometry = TensorProductGrid(
            bounds=[(0, 1), (0, 1), (0, 1)],
            Nx_points=[10, 10, 10],
            boundary_conditions=no_flux_bc(dimension=3),
        )
        geometry.mark_region("box", predicate=lambda x: np.all(x < 0.5, axis=1))

        bc_config = {"box": BCSegment(name="box_bc", bc_type=BCType.DIRICHLET, value=0.0)}

        bc = mixed_bc_from_regions(geometry, bc_config)

        assert bc.dimension == 3  # Inferred from geometry

    def test_validates_geometry_supports_region_marking(self):
        """Test error when geometry doesn't support SupportsRegionMarking."""
        import pytest

        from mfgarchon.geometry.boundary import mixed_bc_from_regions

        # Mock geometry without SupportsRegionMarking
        class FakeGeometry:
            dimension = 2

        fake_geometry = FakeGeometry()
        bc_config = {"region1": BCSegment(name="bc1", bc_type=BCType.DIRICHLET)}

        with pytest.raises(TypeError, match="SupportsRegionMarking"):
            mixed_bc_from_regions(fake_geometry, bc_config)

    def test_validates_region_exists(self):
        """Test error when region doesn't exist in geometry."""
        import pytest

        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import mixed_bc_from_regions

        geometry = TensorProductGrid(
            bounds=[(0, 1), (0, 1)], Nx_points=[50, 50], boundary_conditions=no_flux_bc(dimension=2)
        )
        geometry.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.1)

        # Try to reference non-existent region
        bc_config = {"nonexistent": BCSegment(name="bad", bc_type=BCType.DIRICHLET)}

        with pytest.raises(ValueError, match="Region 'nonexistent' not found"):
            mixed_bc_from_regions(geometry, bc_config)

    def test_default_bc_optional(self):
        """Test that default BC is optional."""
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import mixed_bc_from_regions

        geometry = TensorProductGrid(
            bounds=[(0, 1), (0, 1)], Nx_points=[50, 50], boundary_conditions=no_flux_bc(dimension=2)
        )
        geometry.mark_region("inlet", predicate=lambda x: x[:, 0] < 0.1)

        # No default BC specified
        bc_config = {"inlet": BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0)}

        bc = mixed_bc_from_regions(geometry, bc_config)

        # Issue #1100: no "default" segment -> factory falls back to safe NO_FLUX
        # (was PERIODIC). The point of this test is that a default is optional.
        assert bc.default_bc == BCType.NO_FLUX
        assert bc.default_value == 0.0

    def test_preserves_segment_fields(self):
        """Test that all segment fields are preserved."""
        from mfgarchon.geometry import TensorProductGrid
        from mfgarchon.geometry.boundary import mixed_bc_from_regions

        geometry = TensorProductGrid(
            bounds=[(0, 1), (0, 1)], Nx_points=[50, 50], boundary_conditions=no_flux_bc(dimension=2)
        )
        geometry.mark_region("special", predicate=lambda x: x[:, 0] < 0.1)

        # Create segment with many fields
        bc_config = {
            "special": BCSegment(
                name="special_bc",
                bc_type=BCType.ROBIN,
                value=2.5,
                alpha=1.5,
                beta=0.5,
                priority=10,
                flux_capacity=100.0,
            )
        }

        bc = mixed_bc_from_regions(geometry, bc_config)

        segment = bc.segments[0]
        assert segment.region_name == "special"
        assert segment.name == "special_bc"
        assert segment.bc_type == BCType.ROBIN
        assert segment.value == 2.5
        assert segment.alpha == 1.5
        assert segment.beta == 0.5
        assert segment.priority == 10
        assert segment.flux_capacity == 100.0


class TestMixedBoundaryConditions:
    """Test MixedBoundaryConditions class."""

    def test_basic_creation(self):
        """Test creating empty mixed BC."""
        mixed_bc = MixedBoundaryConditions(dimension=2)
        assert mixed_bc.dimension == 2
        assert len(mixed_bc.segments) == 0
        # Issue #1100: default_bc is None ("unspecified"), no longer implicit PERIODIC.
        assert mixed_bc.default_bc is None
        assert mixed_bc.default_value == 0.0

    def test_segment_sorting_by_priority(self):
        """Test that segments are sorted by priority (highest first)."""
        low_priority = BCSegment(name="low", bc_type=BCType.NEUMANN, priority=0)
        high_priority = BCSegment(name="high", bc_type=BCType.DIRICHLET, priority=10)
        medium_priority = BCSegment(name="medium", bc_type=BCType.ROBIN, priority=5)

        mixed_bc = MixedBoundaryConditions(
            dimension=1,
            segments=[low_priority, high_priority, medium_priority],
        )

        # Should be sorted: high (10), medium (5), low (0)
        assert mixed_bc.segments[0].name == "high"
        assert mixed_bc.segments[1].name == "medium"
        assert mixed_bc.segments[2].name == "low"

    def test_protocol_v14_example(self):
        """Test Protocol v1.4 mixed BC (2D crowd motion)."""
        # Exit on right wall, y in [4.25, 5.75] (absorbing)
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="right",
            region={"y": (4.25, 5.75)},
            priority=1,
        )

        # Walls everywhere else (reflecting)
        wall_bc = BCSegment(
            name="walls",
            bc_type=BCType.NEUMANN,
            value=0.0,
            priority=0,
        )

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc, wall_bc],
            domain_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )

        # Point on exit should get Dirichlet BC
        exit_point = np.array([10.0, 5.0])
        bc_at_exit = mixed_bc.get_bc_at_point(exit_point, "right")
        assert bc_at_exit.bc_type == BCType.DIRICHLET
        assert bc_at_exit.name == "exit"

        # Point on right wall outside exit should get Neumann BC
        wall_point = np.array([10.0, 2.0])
        bc_at_wall = mixed_bc.get_bc_at_point(wall_point, "right")
        assert bc_at_wall.bc_type == BCType.NEUMANN
        assert bc_at_wall.name == "walls"

        # Point on left wall should get Neumann BC
        left_wall_point = np.array([0.0, 5.0])
        bc_at_left = mixed_bc.get_bc_at_point(left_wall_point, "left")
        assert bc_at_left.bc_type == BCType.NEUMANN
        assert bc_at_left.name == "walls"

    def test_get_bc_at_point_uses_highest_priority(self):
        """Test that highest priority segment is selected when multiple match."""
        # General wall BC (low priority)
        general_bc = BCSegment(
            name="general",
            bc_type=BCType.NEUMANN,
            value=0.0,
            boundary="all",
            priority=0,
        )

        # Specific exit BC (high priority)
        specific_bc = BCSegment(
            name="specific",
            bc_type=BCType.DIRICHLET,
            value=1.0,
            boundary="right",
            region={"y": (4.0, 6.0)},
            priority=10,
        )

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[general_bc, specific_bc],
            domain_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )

        # Point that matches both segments should get higher priority
        point = np.array([10.0, 5.0])
        bc = mixed_bc.get_bc_at_point(point, "right")
        assert bc.name == "specific"
        assert bc.bc_type == BCType.DIRICHLET
        assert bc.priority == 10

    def test_get_bc_at_point_fallback_to_default(self):
        """Test fallback to default BC when no segment matches."""
        segment = BCSegment(
            name="specific",
            bc_type=BCType.DIRICHLET,
            boundary="right",
            region={"y": (4.0, 6.0)},
        )

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[segment],
            default_bc=BCType.PERIODIC,
            default_value=5.0,
            domain_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )

        # Point that doesn't match specific segment gets default
        point = np.array([0.0, 2.0])
        bc = mixed_bc.get_bc_at_point(point, "left")
        assert bc.bc_type == BCType.PERIODIC
        assert bc.get_value(point) == 5.0
        assert bc.name == "default"

    def test_identify_boundary_id_2d(self):
        """Test automatic boundary identification in 2D."""
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            domain_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )

        # Test all four boundaries
        assert mixed_bc.identify_boundary_id(np.array([0.0, 5.0])) == "x_min"
        assert mixed_bc.identify_boundary_id(np.array([10.0, 5.0])) == "x_max"
        assert mixed_bc.identify_boundary_id(np.array([5.0, 0.0])) == "y_min"
        assert mixed_bc.identify_boundary_id(np.array([5.0, 10.0])) == "y_max"

        # Interior point should return None
        assert mixed_bc.identify_boundary_id(np.array([5.0, 5.0])) is None

    def test_validate_success(self):
        """Test validation of valid configuration."""
        segment1 = BCSegment(name="seg1", bc_type=BCType.DIRICHLET, priority=1)
        segment2 = BCSegment(name="seg2", bc_type=BCType.NEUMANN, priority=2)

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[segment1, segment2],
            domain_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )

        is_valid, warnings = mixed_bc.validate()
        assert is_valid
        assert len(warnings) == 0

    def test_validate_dimension_mismatch(self):
        """Test validation catches dimension mismatch."""
        # Segment with region for axis 3 in 2D problem
        segment = BCSegment(
            name="bad",
            bc_type=BCType.DIRICHLET,
            region={3: (0, 1)},  # Axis 3 doesn't exist in 2D
        )

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[segment],
            domain_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )

        is_valid, warnings = mixed_bc.validate()
        assert not is_valid
        assert len(warnings) > 0
        assert "dimension" in warnings[0].lower()

    def test_validate_priority_conflict(self):
        """Test validation warns about same-priority segments."""
        seg1 = BCSegment(name="seg1", bc_type=BCType.DIRICHLET, priority=5)
        seg2 = BCSegment(name="seg2", bc_type=BCType.NEUMANN, priority=5)

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[seg1, seg2],
            domain_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )

        is_valid, warnings = mixed_bc.validate()
        assert not is_valid
        assert len(warnings) > 0
        assert "priority" in warnings[0].lower()

    def test_string_representation(self):
        """Test string representation of mixed BC."""
        segment = BCSegment(name="test", bc_type=BCType.DIRICHLET, value=0.0)
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[segment],
            default_bc=BCType.PERIODIC,
        )

        s = str(mixed_bc)
        # Class is now named BoundaryConditions (MixedBoundaryConditions is alias)
        assert "BoundaryConditions" in s
        assert "2D" in s
        assert "test" in s or "dirichlet" in s.lower()  # Either segment name or type


class TestDimensionAgnosticism:
    """Test that mixed BC works across dimensions."""

    def test_1d_problem(self):
        """Test mixed BC for 1D problem."""
        left_bc = BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0, boundary="x_min")
        right_bc = BCSegment(name="right", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max")

        mixed_bc = MixedBoundaryConditions(
            dimension=1,
            segments=[left_bc, right_bc],
            domain_bounds=np.array([[0.0, 1.0]]),
        )

        # Check left boundary
        bc_left = mixed_bc.get_bc_at_point(np.array([0.0]), "x_min")
        assert bc_left.bc_type == BCType.DIRICHLET

        # Check right boundary
        bc_right = mixed_bc.get_bc_at_point(np.array([1.0]), "x_max")
        assert bc_right.bc_type == BCType.NEUMANN

    def test_3d_problem(self):
        """Test mixed BC for 3D problem."""
        segment = BCSegment(
            name="outlet",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            region={"x": (9, 10), "y": (4, 6), "z": (4, 6)},
        )

        mixed_bc = MixedBoundaryConditions(
            dimension=3,
            segments=[segment],
            domain_bounds=np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]),
            # Issue #1100: fall-through default must be explicit now. This test
            # exercises the default-segment dispatch, so set it explicitly.
            default_bc=BCType.PERIODIC,
        )

        # Point in outlet region
        outlet_point = np.array([9.5, 5.0, 5.0])
        bc = mixed_bc.get_bc_at_point(outlet_point, "x_max")
        assert bc.bc_type == BCType.DIRICHLET

        # Point outside outlet region -> falls through to the explicit default
        wall_point = np.array([9.5, 2.0, 5.0])
        bc_wall = mixed_bc.get_bc_at_point(wall_point, "x_max")
        assert bc_wall.bc_type == BCType.PERIODIC  # explicit default


class TestSDFBasedBC:
    """Tests for SDF-based boundary conditions (general domains)."""

    def test_sdf_region_matching(self):
        """Test BC segment matching using SDF region."""
        # Create a segment that applies only near a specific point
        corner_bc = BCSegment(
            name="corner",
            bc_type=BCType.DIRICHLET,
            value=1.0,
            sdf_region=lambda x: np.linalg.norm(x - np.array([1.0, 1.0])) - 0.3,
            priority=2,
        )

        default_bc = BCSegment(
            name="default",
            bc_type=BCType.NEUMANN,
            value=0.0,
            priority=0,
        )

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[corner_bc, default_bc],
            domain_bounds=np.array([[0.0, 2.0], [0.0, 2.0]]),
        )

        # Point inside SDF region (near corner)
        point_in = np.array([1.1, 1.1])
        bc = mixed_bc.get_bc_at_point(point_in, "x_max")
        assert bc.name == "corner"
        assert bc.bc_type == BCType.DIRICHLET

        # Point outside SDF region
        point_out = np.array([0.5, 0.5])
        bc_out = mixed_bc.get_bc_at_point(point_out, "x_max")
        assert bc_out.name == "default"
        assert bc_out.bc_type == BCType.NEUMANN

    def test_normal_direction_matching(self):
        """Test BC segment matching using normal direction."""

        # Define a circular domain SDF
        def circle_sdf(x):
            return np.linalg.norm(np.asarray(x) - np.array([0.0, 0.0])) - 5.0

        # Exit at top (normal pointing up)
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            normal_direction=np.array([0.0, 1.0]),
            normal_tolerance=0.7,  # ~45 degrees
            priority=1,
        )

        # Walls everywhere else
        wall_bc = BCSegment(
            name="walls",
            bc_type=BCType.NEUMANN,
            value=0.0,
            priority=0,
        )

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc, wall_bc],
            domain_sdf=circle_sdf,
        )

        # Point at top of circle (y=5, normal points up)
        top_point = np.array([0.0, 5.0])
        bc = mixed_bc.get_bc_at_point(top_point)
        assert bc.name == "exit"
        assert bc.bc_type == BCType.DIRICHLET

        # Point at right of circle (normal points right, not up)
        right_point = np.array([5.0, 0.0])
        bc_right = mixed_bc.get_bc_at_point(right_point)
        assert bc_right.name == "walls"
        assert bc_right.bc_type == BCType.NEUMANN

    def test_sdf_boundary_identification(self):
        """Test boundary identification for SDF domains."""

        def circle_sdf(x):
            return np.linalg.norm(np.asarray(x) - np.array([0.0, 0.0])) - 5.0

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[],
            domain_sdf=circle_sdf,
        )

        # Point at top (y_max)
        assert mixed_bc.identify_boundary_id(np.array([0.0, 5.0])) == "y_max"

        # Point at right (x_max)
        assert mixed_bc.identify_boundary_id(np.array([5.0, 0.0])) == "x_max"

        # Point at bottom (y_min)
        assert mixed_bc.identify_boundary_id(np.array([0.0, -5.0])) == "y_min"

        # Point at left (x_min)
        assert mixed_bc.identify_boundary_id(np.array([-5.0, 0.0])) == "x_min"

        # Interior point should return None
        assert mixed_bc.identify_boundary_id(np.array([0.0, 0.0])) is None

    def test_is_on_boundary_sdf(self):
        """Test is_on_boundary for SDF domains."""

        def circle_sdf(x):
            return np.linalg.norm(np.asarray(x) - np.array([0.0, 0.0])) - 5.0

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[],
            domain_sdf=circle_sdf,
        )

        # Point on boundary
        assert mixed_bc.is_on_boundary(np.array([5.0, 0.0]))
        assert mixed_bc.is_on_boundary(np.array([0.0, 5.0]))

        # Interior point
        assert not mixed_bc.is_on_boundary(np.array([0.0, 0.0]))

        # Exterior point
        assert not mixed_bc.is_on_boundary(np.array([10.0, 0.0]))

    def test_get_outward_normal_sdf(self):
        """Test outward normal computation for SDF domains."""

        def circle_sdf(x):
            return np.linalg.norm(np.asarray(x) - np.array([0.0, 0.0])) - 5.0

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[],
            domain_sdf=circle_sdf,
        )

        # Normal at top should point up
        normal_top = mixed_bc.get_outward_normal(np.array([0.0, 5.0]))
        assert normal_top is not None
        assert np.allclose(normal_top, np.array([0.0, 1.0]), atol=1e-3)

        # Normal at right should point right
        normal_right = mixed_bc.get_outward_normal(np.array([5.0, 0.0]))
        assert normal_right is not None
        assert np.allclose(normal_right, np.array([1.0, 0.0]), atol=1e-3)

    def test_validate_normal_direction_dimension(self):
        """Test validation catches wrong normal direction dimension."""
        segment = BCSegment(
            name="bad",
            bc_type=BCType.DIRICHLET,
            normal_direction=np.array([1.0, 0.0, 0.0]),  # 3D in 2D problem
        )

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[segment],
            domain_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        )

        is_valid, warnings = mixed_bc.validate()
        assert not is_valid
        assert any("normal_direction" in w for w in warnings)

    def test_combined_sdf_region_and_normal(self):
        """Test combining SDF region and normal direction constraints."""

        def circle_sdf(x):
            return np.linalg.norm(np.asarray(x) - np.array([0.0, 0.0])) - 5.0

        # Exit only at top-right quadrant
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            normal_direction=np.array([0.0, 1.0]),  # Normal pointing up
            normal_tolerance=0.7,
            sdf_region=lambda x: -x[0],  # Only for x > 0
            priority=1,
        )

        wall_bc = BCSegment(
            name="walls",
            bc_type=BCType.NEUMANN,
            value=0.0,
            priority=0,
        )

        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc, wall_bc],
            domain_sdf=circle_sdf,
        )

        # Top-right point (x>0, normal up) - should match exit
        # Position at ~45 degrees: (3.54, 3.54) is approximately on circle boundary
        top_right = np.array([3.54, 3.54])
        bc = mixed_bc.get_bc_at_point(top_right)
        assert bc.name == "exit"

        # Top-left point (x<0, normal up) - should NOT match exit (fails sdf_region)
        top_left = np.array([-3.54, 3.54])
        bc_left = mixed_bc.get_bc_at_point(top_left)
        assert bc_left.name == "walls"

    def test_string_representation_with_sdf(self):
        """Test string representation includes SDF info."""
        segment = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            sdf_region=lambda x: x[0] - 1.0,
            normal_direction=np.array([1.0, 0.0]),
        )

        s = str(segment)
        assert "sdf_region" in s
        assert "normal" in s

        # Test MixedBoundaryConditions string shows SDF domain type
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[segment],
            domain_sdf=lambda x: np.linalg.norm(x) - 1.0,
        )

        s_mixed = str(mixed_bc)
        assert "SDF" in s_mixed
