"""
Unit tests for BC applicator utilities.

Tests the application of uniform and mixed boundary conditions to grid fields.
"""

import pytest

import numpy as np

from mfgarchon.geometry.boundary import (
    BCSegment,
    BCType,
    MixedBoundaryConditions,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    # Factory functions for creating BCs
    periodic_bc,
)

# Import deprecated functions from _compat for legacy testing
# These are no longer publicly exported (Issue #577 Phase 3)
from mfgarchon.geometry.boundary._compat import (
    apply_boundary_conditions_2d,
    apply_boundary_conditions_nd,
    create_boundary_mask_2d,
)
from mfgarchon.geometry.boundary.applicator_fdm import GhostCellConfig


class TestUniformBC2D:
    """Tests for uniform BC application in 2D."""

    def test_periodic_bc(self):
        """Test periodic boundary conditions."""
        field = np.arange(25).reshape(5, 5).astype(float)
        bc = periodic_bc(dimension=2)
        padded = apply_boundary_conditions_2d(field, bc)

        assert padded.shape == (7, 7)
        # Check wrap-around
        assert padded[0, 3] == field[-1, 2]  # Top ghost from bottom interior
        assert padded[-1, 3] == field[0, 2]  # Bottom ghost from top interior
        assert padded[3, 0] == field[2, -1]  # Left ghost from right interior
        assert padded[3, -1] == field[2, 0]  # Right ghost from left interior

    def test_dirichlet_bc(self):
        """Test Dirichlet boundary conditions."""
        field = np.ones((5, 5))
        bc = dirichlet_bc(value=0.0, dimension=2)
        padded = apply_boundary_conditions_2d(field, bc)

        assert padded.shape == (7, 7)
        # Ghost cell formula for cell-centered grid: ghost = 2*g - interior
        # With g=0 and interior=1: ghost = 2*0 - 1 = -1
        assert padded[0, 3] == -1.0
        assert padded[-1, 3] == -1.0
        assert padded[3, 0] == -1.0
        assert padded[3, -1] == -1.0

    def test_neumann_bc(self):
        """Test Neumann (no-flux) boundary conditions."""
        field = np.arange(25).reshape(5, 5).astype(float)
        bc = neumann_bc(dimension=2)
        padded = apply_boundary_conditions_2d(field, bc)

        assert padded.shape == (7, 7)
        # Issue #542: Neumann uses reflection (ghost = next interior, not adjacent)
        # This gives O(h²) accurate ghost values for zero-flux BC
        # padded[0, :] = padded[2, :] (reflect about index 1)
        assert padded[0, 3] == field[1, 2]  # Low ghost = reflected (next interior row)
        assert padded[-1, 3] == field[-2, 2]  # High ghost = reflected (prev interior row)
        assert padded[3, 0] == field[2, 1]  # Left ghost = reflected (next interior col)
        assert padded[3, -1] == field[2, -2]  # Right ghost = reflected (prev interior col)

    def test_no_flux_bc(self):
        """Test no_flux alias for Neumann BC."""
        field = np.ones((5, 5))
        bc = no_flux_bc(dimension=2)
        padded = apply_boundary_conditions_2d(field, bc)

        assert padded.shape == (7, 7)
        # All ghost cells should equal 1 (edge padding of uniform field)
        assert np.allclose(padded, 1.0)


class TestMixedBC2D:
    """Tests for mixed BC application in 2D."""

    def test_basic_mixed_bc(self):
        """Test basic mixed BC with exit and walls."""
        # Exit on right wall (Dirichlet), walls elsewhere (Neumann)
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="x_max",
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
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))
        padded = apply_boundary_conditions_2d(field, mixed_bc)

        assert padded.shape == (7, 7)

        # Right boundary (exit, Dirichlet): ghost = 2*g - interior = -1
        # Interior is column -2 in padded = column 4 in field = 1.0
        assert np.allclose(padded[1:-1, -1], -1.0)

        # Left boundary (walls, Neumann): ghost = interior = 1
        assert np.allclose(padded[1:-1, 0], 1.0)

    def test_partial_exit(self):
        """Test mixed BC with exit on part of boundary."""
        # Exit on right wall, y in [0.4, 0.6] (Dirichlet)
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="x_max",
            region={"y": (0.4, 0.6)},
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
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))
        padded = apply_boundary_conditions_2d(field, mixed_bc)

        # y coordinates: [0.0, 0.25, 0.5, 0.75, 1.0]
        # Exit region y in [0.4, 0.6] covers only y=0.5 (index 2)

        # Right ghost column: Dirichlet at y=0.5, Neumann elsewhere
        # y=0.0 (padded row 1): Neumann -> ghost = interior = 1
        # y=0.25 (padded row 2): Neumann -> ghost = interior = 1
        # y=0.5 (padded row 3): Dirichlet -> ghost = 2*0 - 1 = -1
        # y=0.75 (padded row 4): Neumann -> ghost = interior = 1
        # y=1.0 (padded row 5): Neumann -> ghost = interior = 1

        assert padded[1, -1] == 1.0  # y=0.0, Neumann
        assert padded[2, -1] == 1.0  # y=0.25, Neumann
        assert padded[3, -1] == -1.0  # y=0.5, Dirichlet
        assert padded[4, -1] == 1.0  # y=0.75, Neumann
        assert padded[5, -1] == 1.0  # y=1.0, Neumann

    def test_domain_bounds_from_mixed_bc(self):
        """Test that domain_bounds can be provided via MixedBoundaryConditions."""
        exit_bc = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0)
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))
        # Should work without passing domain_bounds explicitly
        padded = apply_boundary_conditions_2d(field, mixed_bc)
        assert padded.shape == (7, 7)


class TestBC1D:
    """Tests for 1D boundary conditions."""

    def test_uniform_bc_1d(self):
        """Test uniform BC in 1D."""
        field = np.ones(5)
        bc = neumann_bc(dimension=1)
        padded = apply_boundary_conditions_nd(field, bc)

        assert padded.shape == (7,)
        assert np.allclose(padded, 1.0)

    def test_dirichlet_bc_1d(self):
        """Test Dirichlet BC in 1D with uniform value."""
        field = np.ones(5)
        # Use uniform Dirichlet with value 0 for simplicity
        bc = dirichlet_bc(value=0.0, dimension=1)
        padded = apply_boundary_conditions_nd(field, bc)

        assert padded.shape == (7,)
        # Ghost cell formula: ghost = 2*g - interior
        # With g=0 and interior=1: ghost = 2*0 - 1 = -1
        assert padded[0] == -1.0
        assert padded[-1] == -1.0

    def test_mixed_bc_1d(self):
        """Test mixed BC in 1D."""
        left_bc = BCSegment(
            name="inlet",
            bc_type=BCType.DIRICHLET,
            value=1.0,
            boundary="x_min",
            priority=1,
        )
        right_bc = BCSegment(
            name="outlet",
            bc_type=BCType.NEUMANN,
            value=0.0,
            boundary="x_max",
            priority=1,
        )
        mixed_bc = MixedBoundaryConditions(
            dimension=1,
            segments=[left_bc, right_bc],
            domain_bounds=np.array([[0.0, 1.0]]),
        )

        field = np.zeros(5)
        padded = apply_boundary_conditions_nd(field, mixed_bc)

        assert padded.shape == (7,)
        # Left: Dirichlet g=1, interior=0 -> ghost = 2*1 - 0 = 2
        assert padded[0] == 2.0
        # Right: Neumann g=0 -> ghost = interior = 0
        assert padded[-1] == 0.0


class TestBoundaryMask:
    """Tests for pre-computed boundary masks."""

    def test_create_boundary_mask(self):
        """Test creation of boundary masks for efficiency."""
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="x_max",
            region={"y": (0.4, 0.6)},
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
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        masks = create_boundary_mask_2d(mixed_bc, (5, 5), mixed_bc.domain_bounds)

        # Check exit mask on right boundary
        assert "exit" in masks
        assert "walls" in masks

        # Exit only on right boundary at y=0.5 (index 2)
        assert masks["exit"]["left"].sum() == 0
        assert masks["exit"]["right"].sum() == 1
        assert masks["exit"]["right"][2]  # y=0.5

        # Walls everywhere else
        assert masks["walls"]["left"].sum() == 5  # All left boundary
        assert masks["walls"]["right"].sum() == 4  # Right except exit


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_domain_bounds_error(self):
        """Test that missing domain_bounds raises error for mixed BC."""
        # Create a proper mixed BC with boundary restrictions (not uniform)
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="x_max",  # This makes it a true mixed BC
            priority=1,
        )
        wall_bc = BCSegment(
            name="wall",
            bc_type=BCType.NEUMANN,
            value=0.0,
            priority=0,
        )
        mixed_bc_obj = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc, wall_bc],
            # No domain_bounds
        )

        field = np.ones((5, 5))
        with pytest.raises(ValueError, match="domain_bounds"):
            apply_boundary_conditions_2d(field, mixed_bc_obj)

    def test_unsupported_bc_type_error(self):
        """Test that invalid BC type raises error during creation."""
        # With the new unified class, invalid types raise error during creation
        from mfgarchon.geometry.boundary import uniform_bc

        with pytest.raises(ValueError):
            uniform_bc(bc_type="unknown_type")

    def test_corner_handling(self):
        """Test that corners are handled correctly (averaged)."""
        exit_bc = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0)
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((3, 3))
        padded = apply_boundary_conditions_2d(field, mixed_bc)

        # Corners should be average of adjacent ghost cells
        # Bottom-left corner
        expected_bl = 0.5 * (padded[0, 1] + padded[1, 0])
        assert padded[0, 0] == expected_bl

        # Top-right corner
        expected_tr = 0.5 * (padded[-1, -2] + padded[-2, -1])
        assert padded[-1, -1] == expected_tr


class TestRobinBC:
    """Tests for Robin boundary condition: alpha*u + beta*du/dn = g."""

    def test_robin_bc_basic(self):
        """Test Robin BC with alpha=1, beta=1, g=0.5."""
        # Robin BC: alpha*u + beta*du/dn = g at boundary
        # Ghost formula: ghost = (2*g - alpha*interior + beta*interior/dx) / (alpha + beta/dx)
        robin_bc = BCSegment(
            name="robin",
            bc_type=BCType.ROBIN,
            value=0.5,  # g = 0.5
            alpha=1.0,
            beta=1.0,
            priority=1,
        )
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[robin_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))
        padded = apply_boundary_conditions_2d(field, mixed_bc)

        # dx = 1.0/4 = 0.25, interior = 1.0, g = 0.5, alpha = 1, beta = 1
        # For x_min boundary (boundary_side="min"):
        #   ghost = (2*0.5 - 1*1 + 1*1/0.25) / (1 + 1/0.25)
        #        = (1 - 1 + 4) / (1 + 4) = 4/5 = 0.8
        # Note: sign changes for min vs max boundary
        assert padded.shape == (7, 7)
        # Ghost cells should be different from interior due to Robin BC

    def test_robin_reduces_to_dirichlet(self):
        """Test that Robin with beta=0 is equivalent to Dirichlet."""
        # When beta=0: alpha*u = g at boundary -> u = g/alpha
        robin_bc = BCSegment(
            name="dirichlet_like",
            bc_type=BCType.ROBIN,
            value=2.0,  # g = 2
            alpha=1.0,
            beta=0.0,  # No flux term -> pure Dirichlet
            priority=1,
        )
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[robin_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))
        padded = apply_boundary_conditions_2d(field, mixed_bc)

        # For Dirichlet-like Robin with g=2, interior=1:
        # ghost = 2*g - interior = 2*2 - 1 = 3
        assert np.isclose(padded[3, 0], 3.0)  # Left boundary
        assert np.isclose(padded[3, -1], 3.0)  # Right boundary

    def test_robin_reduces_to_neumann(self):
        """Test that Robin with alpha=0 is equivalent to Neumann."""
        # When alpha=0: beta*du/dn = g -> du/dn = g/beta
        robin_bc = BCSegment(
            name="neumann_like",
            bc_type=BCType.ROBIN,
            value=0.0,  # g = 0 (no-flux)
            alpha=0.0,
            beta=1.0,  # Only flux term -> pure Neumann
            priority=1,
        )
        mixed_bc = MixedBoundaryConditions(
            dimension=2,
            segments=[robin_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.arange(25).reshape(5, 5).astype(float)
        padded = apply_boundary_conditions_2d(field, mixed_bc)

        # Issue #542: Neumann uses reflection (ghost = next interior, not adjacent)
        # For zero-flux (g=0): ghost = interior_next to maintain O(h²) accuracy
        assert np.isclose(padded[3, 0], field[2, 1])  # Left: ghost = next interior (reflected)
        assert np.isclose(padded[3, -1], field[2, -2])  # Right: ghost = next interior (reflected)


class TestTimeDependentBC:
    """Tests for time-dependent boundary values."""

    def test_time_dependent_dirichlet(self):
        """Test Dirichlet BC with time-varying value."""

        # Value depends on time: g(t) = 2*t
        def time_varying_value(point, time):
            return 2.0 * time

        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=time_varying_value,
            boundary="x_max",
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
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))

        # At t=0: g=0, ghost = 2*0 - 1 = -1
        padded_t0 = apply_boundary_conditions_2d(field, mixed_bc, time=0.0)
        assert np.allclose(padded_t0[1:-1, -1], -1.0)

        # At t=1: g=2, ghost = 2*2 - 1 = 3
        padded_t1 = apply_boundary_conditions_2d(field, mixed_bc, time=1.0)
        assert np.allclose(padded_t1[1:-1, -1], 3.0)

        # At t=0.5: g=1, ghost = 2*1 - 1 = 1
        padded_t05 = apply_boundary_conditions_2d(field, mixed_bc, time=0.5)
        assert np.allclose(padded_t05[1:-1, -1], 1.0)

    def test_space_and_time_dependent_neumann(self):
        """Test Neumann BC with value depending on both space and time."""

        # Flux = x * t (position-dependent flux that varies with time)
        def space_time_flux(point, time):
            return point[0] * time

        flux_bc = BCSegment(
            name="varying_flux",
            bc_type=BCType.NEUMANN,
            value=space_time_flux,
            boundary="y_max",  # Top boundary
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
            segments=[flux_bc, wall_bc],
            domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

        field = np.ones((5, 5))

        # At t=0: flux=0 everywhere, ghost = interior
        padded_t0 = apply_boundary_conditions_2d(field, mixed_bc, time=0.0)
        assert np.isclose(padded_t0[-1, 3], 1.0)  # No change from interior

        # At t=1: flux varies with x position
        # Top boundary points have x coords [0, 0.25, 0.5, 0.75, 1.0]
        # At x=0.5 (center): flux = 0.5*1 = 0.5
        # ghost = interior + 2*dx*flux = 1 + 2*0.25*0.5 = 1.25
        _padded_t1 = apply_boundary_conditions_2d(field, mixed_bc, time=1.0)
        # The ghost values at top should vary with x (stored in _padded_t1 for future assertions)


class TestGridTypeConfiguration:
    """Tests for cell-centered vs vertex-centered grid configuration."""

    def test_cell_centered_dirichlet(self):
        """Test Dirichlet BC with cell-centered grid (default)."""
        field = np.ones((5, 5))
        bc = dirichlet_bc(value=0.0, dimension=2)

        config = GhostCellConfig(grid_type="cell_centered")
        padded = apply_boundary_conditions_2d(field, bc, config=config)

        # Cell-centered: ghost = 2*g - interior = 2*0 - 1 = -1
        assert np.isclose(padded[3, 0], -1.0)
        assert np.isclose(padded[3, -1], -1.0)

    def test_vertex_centered_dirichlet(self):
        """Test Dirichlet BC with vertex-centered grid."""
        field = np.ones((5, 5))
        bc = dirichlet_bc(value=0.0, dimension=2)

        config = GhostCellConfig(grid_type="vertex_centered")
        padded = apply_boundary_conditions_2d(field, bc, config=config)

        # Vertex-centered: ghost = g (boundary value at vertex)
        assert np.isclose(padded[3, 0], 0.0)
        assert np.isclose(padded[3, -1], 0.0)

    def test_grid_type_enum(self):
        """Test using GridType enum directly (not string)."""
        from mfgarchon.geometry.boundary import GridType

        field = np.ones((5, 5))
        bc = dirichlet_bc(value=0.0, dimension=2)

        # Using enum directly
        config = GhostCellConfig(grid_type=GridType.VERTEX_CENTERED)
        padded = apply_boundary_conditions_2d(field, bc, config=config)

        # Vertex-centered: ghost = g
        assert np.isclose(padded[3, 0], 0.0)

        # Test cell-centered with enum
        config2 = GhostCellConfig(grid_type=GridType.CELL_CENTERED)
        padded2 = apply_boundary_conditions_2d(field, bc, config=config2)

        # Cell-centered: ghost = 2*g - interior = -1
        assert np.isclose(padded2[3, 0], -1.0)

    def test_vertex_centered_neumann(self):
        """Test Neumann BC with vertex-centered grid."""
        field = np.ones((5, 5))
        bc = neumann_bc(value=0.0, dimension=2)

        config = GhostCellConfig(grid_type="vertex_centered")
        padded = apply_boundary_conditions_2d(field, bc, config=config)

        # Neumann with zero flux: ghost = interior (for both grid types)
        assert np.isclose(padded[3, 0], 1.0)
        assert np.isclose(padded[3, -1], 1.0)


class TestExtrapolationBC:
    """Tests for extrapolation boundary conditions (for unbounded domains)."""

    def test_linear_extrapolation_function(self):
        """Test ghost_cell_linear_extrapolation function directly."""
        from mfgarchon.geometry.boundary import ghost_cell_linear_extrapolation

        # Linear function: f(x) = 2x + 1
        # At x=0: f(0)=1, at x=1: f(1)=3
        # Extrapolated to x=-1: f(-1) = -1
        u_0, u_1 = 1.0, 3.0  # Note: u_0 is at boundary, u_1 is one step inside
        # For left boundary extrapolation: ghost = 2*u_0 - u_1 = 2*1 - 3 = -1
        ghost = ghost_cell_linear_extrapolation((u_0, u_1))
        assert np.isclose(ghost, -1.0)

    def test_quadratic_extrapolation_function(self):
        """Test ghost_cell_quadratic_extrapolation function directly."""
        from mfgarchon.geometry.boundary import ghost_cell_quadratic_extrapolation

        # Quadratic function: f(x) = x^2
        # At x=0: f(0)=0, x=1: f(1)=1, x=2: f(2)=4
        # Extrapolated to x=-1: f(-1) = 1
        u_0, u_1, u_2 = 0.0, 1.0, 4.0
        # For left boundary: ghost = 3*u_0 - 3*u_1 + u_2 = 0 - 3 + 4 = 1
        ghost = ghost_cell_quadratic_extrapolation((u_0, u_1, u_2))
        assert np.isclose(ghost, 1.0)

    def test_linear_extrapolation_1d(self):
        """Test linear extrapolation BC in 1D via apply_boundary_conditions_1d."""
        from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
        from mfgarchon.geometry.boundary._compat import apply_boundary_conditions_1d

        # Create a linear field: f(x) = x, on [0, 1] with 5 points
        # x = [0, 0.25, 0.5, 0.75, 1.0]
        # f = [0, 0.25, 0.5, 0.75, 1.0]
        field = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        # Use linear extrapolation on both ends
        bc = BoundaryConditions(
            dimension=1,
            segments=[
                BCSegment(name="left", bc_type=BCType.EXTRAPOLATION_LINEAR, boundary="x_min"),
                BCSegment(name="right", bc_type=BCType.EXTRAPOLATION_LINEAR, boundary="x_max"),
            ],
            domain_bounds=np.array([[0.0, 1.0]]),
        )

        padded = apply_boundary_conditions_1d(field, bc)

        # Left ghost: 2*field[0] - field[1] = 2*0 - 0.25 = -0.25
        assert np.isclose(padded[0], -0.25)
        # Right ghost: 2*field[-1] - field[-2] = 2*1.0 - 0.75 = 1.25
        assert np.isclose(padded[-1], 1.25)

    def test_quadratic_extrapolation_1d(self):
        """Test quadratic extrapolation BC in 1D."""
        from mfgarchon.geometry.boundary import BCSegment, BCType, BoundaryConditions
        from mfgarchon.geometry.boundary._compat import apply_boundary_conditions_1d

        # Create a quadratic field: f(x) = x^2, on [0, 1] with 5 points
        # x = [0, 0.25, 0.5, 0.75, 1.0]
        # f = [0, 0.0625, 0.25, 0.5625, 1.0]
        x = np.linspace(0, 1, 5)
        field = x**2

        # Use quadratic extrapolation on both ends
        bc = BoundaryConditions(
            dimension=1,
            segments=[
                BCSegment(name="left", bc_type=BCType.EXTRAPOLATION_QUADRATIC, boundary="x_min"),
                BCSegment(name="right", bc_type=BCType.EXTRAPOLATION_QUADRATIC, boundary="x_max"),
            ],
            domain_bounds=np.array([[0.0, 1.0]]),
        )

        padded = apply_boundary_conditions_1d(field, bc)

        # Left ghost: 3*f[0] - 3*f[1] + f[2]
        expected_left = 3 * field[0] - 3 * field[1] + field[2]
        assert np.isclose(padded[0], expected_left)

        # Right ghost: 3*f[-1] - 3*f[-2] + f[-3]
        expected_right = 3 * field[-1] - 3 * field[-2] + field[-3]
        assert np.isclose(padded[-1], expected_right)


class TestInputValidation:
    """Tests for input validation."""

    def test_nan_detection(self):
        """Test that NaN values in field are detected."""
        field = np.ones((5, 5))
        field[2, 2] = np.nan

        bc = neumann_bc(dimension=2)
        with pytest.raises(ValueError, match="NaN"):
            apply_boundary_conditions_2d(field, bc)

    def test_inf_detection(self):
        """Test that infinite values in field are detected."""
        field = np.ones((5, 5))
        field[2, 2] = np.inf

        bc = neumann_bc(dimension=2)
        with pytest.raises(ValueError, match="NaN or Inf"):
            apply_boundary_conditions_2d(field, bc)

    def test_invalid_domain_bounds(self):
        """Test that invalid domain bounds are rejected."""
        # Create proper mixed BC with boundary restrictions
        exit_bc = BCSegment(
            name="exit",
            bc_type=BCType.DIRICHLET,
            value=0.0,
            boundary="x_max",
            priority=1,
        )
        wall_bc = BCSegment(
            name="wall",
            bc_type=BCType.NEUMANN,
            value=0.0,
            priority=0,
        )
        # min > max is invalid
        mixed_bc_obj = MixedBoundaryConditions(
            dimension=2,
            segments=[exit_bc, wall_bc],
            domain_bounds=np.array([[1.0, 0.0], [0.0, 1.0]]),  # x_min > x_max
        )

        field = np.ones((5, 5))
        with pytest.raises(ValueError, match="min < max"):
            apply_boundary_conditions_2d(field, mixed_bc_obj)


class TestLazyDimensionBinding:
    """Tests for lazy dimension binding (Issue #495)."""

    def test_dirichlet_bc_no_dimension(self):
        """Test creating Dirichlet BC without dimension."""
        bc = dirichlet_bc(value=0.0)  # No dimension specified
        assert bc.dimension is None
        assert not bc.is_bound
        assert str(bc) == "BoundaryConditions(unbound, dirichlet, value=0.0)"

    def test_neumann_bc_no_dimension(self):
        """Test creating Neumann BC without dimension."""
        bc = neumann_bc(value=0.0)  # No dimension specified
        assert bc.dimension is None
        assert not bc.is_bound

    def test_periodic_bc_no_dimension(self):
        """Test creating Periodic BC without dimension."""
        bc = periodic_bc()  # No dimension specified
        assert bc.dimension is None
        assert not bc.is_bound

    def test_bind_dimension_explicit(self):
        """Test explicit dimension binding via bind_dimension()."""
        bc = dirichlet_bc(value=0.0)
        assert bc.dimension is None

        bc_2d = bc.bind_dimension(2)
        assert bc_2d.dimension == 2
        assert bc_2d.is_bound
        assert str(bc_2d) == "BoundaryConditions(2D, dirichlet, value=0.0)"

        # Original BC should be unchanged (immutable via replace)
        assert bc.dimension is None

    def test_bind_dimension_idempotent(self):
        """Test that binding same dimension twice returns same BC."""
        bc = dirichlet_bc(value=0.0, dimension=2)
        bc_bound = bc.bind_dimension(2)

        # Should return same object since dimension already matches
        assert bc_bound is bc

    def test_bind_dimension_mismatch_error(self):
        """Test that binding different dimension raises error."""
        bc = dirichlet_bc(value=0.0, dimension=2)

        with pytest.raises(ValueError, match="BC dimension mismatch"):
            bc.bind_dimension(3)

    def test_grid_binds_dimension_automatically(self):
        """Test that TensorProductGrid automatically binds dimension."""
        from mfgarchon.geometry import TensorProductGrid

        # Create BC without dimension
        bc = dirichlet_bc(value=0.0)
        assert bc.dimension is None

        # Create grid with BC - dimension should be bound automatically
        grid = TensorProductGrid(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            Nx=[10, 10],
            boundary_conditions=bc,
        )

        # Grid's stored BC should have dimension bound
        stored_bc = grid.get_boundary_conditions()
        assert stored_bc.dimension == 2
        assert stored_bc.is_bound

    def test_unbound_bc_cannot_apply(self):
        """Test that unbound BC cannot be applied directly."""
        bc = dirichlet_bc(value=0.0)  # No dimension

        # Applicator requires dimension to compute ghost cells
        with pytest.raises(ValueError, match="BC dimension not set"):
            # This should fail because dimension is needed for ghost cell computation
            bc._require_dimension("apply BC")

    def test_explicit_dimension_still_works(self):
        """Test that explicit dimension specification still works."""
        bc = dirichlet_bc(value=0.0, dimension=2)
        assert bc.dimension == 2
        assert bc.is_bound

        # Should work with apply_boundary_conditions_2d
        field = np.ones((5, 5))
        padded = apply_boundary_conditions_2d(field, bc)
        assert padded.shape == (7, 7)

    def test_validate_unbound_bc_warns(self):
        """Test that validate() warns about unbound dimension."""
        bc = dirichlet_bc(value=0.0)
        _is_valid, warnings = bc.validate()

        # Should have warning about dimension not set
        assert any("Dimension not set" in w for w in warnings)

    def test_backward_compatibility_dimension_default(self):
        """Test backward compatibility with code that didn't specify dimension."""
        # Before this change, dimension defaulted to 2
        # Now dimension defaults to None, but existing code that passes dimension=2
        # should still work identically
        bc_explicit = dirichlet_bc(value=0.0, dimension=2)
        assert bc_explicit.dimension == 2

        # Code that uses the BC with a grid should work
        field = np.ones((5, 5))
        padded = apply_boundary_conditions_2d(field, bc_explicit)
        assert padded.shape == (7, 7)


class TestSDFParticleBCHandler:
    """Tests for SDF-based particle BC handler (Issue #497)."""

    def test_sdf_handler_creation(self):
        """Test creating an SDF particle BC handler."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_sphere

        def circle_sdf(points):
            return sdf_sphere(points, center=[0, 0], radius=1.0)

        handler = SDFParticleBCHandler(circle_sdf, dimension=2)
        assert handler.dimension == 2
        assert handler.sdf is not None

    def test_particles_inside_unchanged(self):
        """Test that particles inside domain are unchanged."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_sphere

        def circle_sdf(points):
            return sdf_sphere(points, center=[0, 0], radius=1.0)

        handler = SDFParticleBCHandler(circle_sdf, dimension=2)

        # Particles inside the unit circle
        X_old = np.array([[0.0, 0.0], [0.3, 0.0], [0.0, 0.5]])
        X_new = np.array([[0.1, 0.0], [0.4, 0.0], [0.0, 0.6]])

        X_result, _ = handler.apply_bc(X_old, X_new)

        # Should be unchanged (all inside)
        np.testing.assert_array_almost_equal(X_result, X_new)

    def test_particle_crossing_reflected(self):
        """Test that particles crossing boundary are reflected."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_sphere

        def circle_sdf(points):
            return sdf_sphere(points, center=[0, 0], radius=1.0)

        handler = SDFParticleBCHandler(circle_sdf, dimension=2)

        # One particle crosses the boundary
        X_old = np.array([[0.9, 0.0]])  # Inside
        X_new = np.array([[1.2, 0.0]])  # Outside

        X_result, _ = handler.apply_bc(X_old, X_new)

        # Should be reflected back inside
        assert handler.sdf(X_result)[0] <= 0, "Reflected particle should be inside"

    def test_velocity_reflection(self):
        """Test that velocity is reflected at boundary."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_sphere

        def circle_sdf(points):
            return sdf_sphere(points, center=[0, 0], radius=1.0)

        handler = SDFParticleBCHandler(circle_sdf, dimension=2)

        # Particle moving outward along x-axis
        X_old = np.array([[0.9, 0.0]])  # Inside
        X_new = np.array([[1.2, 0.0]])  # Outside
        V = np.array([[1.0, 0.0]])  # Velocity toward boundary

        _X_result, V_result = handler.apply_bc(X_old, X_new, velocities=V)

        # Normal at x=1, y=0 is (1, 0), so velocity should reverse
        assert V_result[0, 0] < 0, "x-velocity should reverse (reflect)"
        np.testing.assert_almost_equal(V_result[0, 1], 0.0)

    def test_contains_method(self):
        """Test the contains() method."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_sphere

        def circle_sdf(points):
            return sdf_sphere(points, center=[0, 0], radius=1.0)

        handler = SDFParticleBCHandler(circle_sdf, dimension=2)

        points = np.array([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]])
        inside = handler.contains(points)

        assert inside[0] is True or inside[0] == True  # noqa: E712 - Center inside
        assert inside[1] is True or inside[1] == True  # noqa: E712 - Mid inside
        assert inside[2] is False or inside[2] == False  # noqa: E712 - Outside

    def test_multiple_particles_mixed(self):
        """Test handling multiple particles with mixed inside/crossing."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_sphere

        def circle_sdf(points):
            return sdf_sphere(points, center=[0, 0], radius=1.0)

        handler = SDFParticleBCHandler(circle_sdf, dimension=2)

        # Mix of inside moves and boundary crossings
        X_old = np.array(
            [
                [0.0, 0.0],  # Stays inside
                [0.9, 0.0],  # Will cross
                [0.0, 0.5],  # Stays inside
            ]
        )
        X_new = np.array(
            [
                [0.1, 0.0],  # Still inside
                [1.2, 0.0],  # Crossed
                [0.0, 0.6],  # Still inside
            ]
        )

        X_result, _ = handler.apply_bc(X_old, X_new)

        # All should end up inside
        sdf_result = handler.sdf(X_result)
        assert np.all(sdf_result <= 0), "All particles should be inside after BC"

        # Non-crossing particles should be unchanged
        np.testing.assert_array_almost_equal(X_result[0], X_new[0])
        np.testing.assert_array_almost_equal(X_result[2], X_new[2])

    def test_box_sdf(self):
        """Test with box/rectangular SDF."""
        from mfgarchon.geometry.boundary import SDFParticleBCHandler
        from mfgarchon.utils.numerical import sdf_box

        def box_sdf(points):
            return sdf_box(points, bounds=[[0, 1], [0, 1]])

        handler = SDFParticleBCHandler(box_sdf, dimension=2)

        # Particle crossing right boundary
        X_old = np.array([[0.9, 0.5]])
        X_new = np.array([[1.2, 0.5]])

        X_result, _ = handler.apply_bc(X_old, X_new)

        # Should be reflected back into box
        assert 0 <= X_result[0, 0] <= 1, "x should be in [0, 1]"
        assert 0 <= X_result[0, 1] <= 1, "y should be in [0, 1]"


# =============================================================================
# Topology/Calculator Composition Tests (Issue #516)
# =============================================================================


class TestTopologyClasses:
    """Test Topology implementations (PeriodicTopology, BoundedTopology)."""

    def test_periodic_topology_creation(self):
        """Test PeriodicTopology initialization."""
        from mfgarchon.geometry.boundary import PeriodicTopology

        topo = PeriodicTopology(dimension=2, shape=(10, 15))
        assert topo.is_periodic is True
        assert topo.dimension == 2
        assert topo.shape == (10, 15)

    def test_bounded_topology_creation(self):
        """Test BoundedTopology initialization."""
        from mfgarchon.geometry.boundary import BoundedTopology

        topo = BoundedTopology(dimension=3, shape=(5, 6, 7))
        assert topo.is_periodic is False
        assert topo.dimension == 3
        assert topo.shape == (5, 6, 7)

    def test_topology_dimension_shape_mismatch_error(self):
        """Test that mismatched dimension and shape raises error."""
        from mfgarchon.geometry.boundary import PeriodicTopology

        with pytest.raises(ValueError, match="Shape length"):
            PeriodicTopology(dimension=2, shape=(10, 15, 20))

    def test_topology_repr(self):
        """Test topology string representation."""
        from mfgarchon.geometry.boundary import BoundedTopology, PeriodicTopology

        periodic = PeriodicTopology(dimension=2, shape=(10, 10))
        assert "PeriodicTopology" in repr(periodic)
        assert "dimension=2" in repr(periodic)

        bounded = BoundedTopology(dimension=2, shape=(10, 10))
        assert "BoundedTopology" in repr(bounded)


class TestCalculatorClasses:
    """Test BoundaryCalculator implementations."""

    def test_dirichlet_calculator(self):
        """Test DirichletCalculator computes correct ghost values."""
        from mfgarchon.geometry.boundary import DirichletCalculator

        calc = DirichletCalculator(boundary_value=5.0)
        # Cell-centered: u_ghost = 2*g - u_interior = 2*5 - 3 = 7
        ghost = calc.compute(interior_value=3.0, dx=0.1, side="min")
        assert np.isclose(ghost, 7.0)

    def test_neumann_calculator_zero_flux(self):
        """Test NeumannCalculator with zero flux (edge extension)."""
        from mfgarchon.geometry.boundary import NeumannCalculator

        calc = NeumannCalculator(flux_value=0.0)
        ghost = calc.compute(interior_value=3.0, dx=0.1, side="min")
        # Zero flux: ghost = interior
        assert np.isclose(ghost, 3.0)

    def test_neumann_calculator_nonzero_flux(self):
        """Test NeumannCalculator with non-zero flux."""
        from mfgarchon.geometry.boundary import NeumannCalculator

        calc = NeumannCalculator(flux_value=1.0)
        dx = 0.1

        # For min side (outward_sign = -1): ghost = interior - 2*dx*g
        ghost_min = calc.compute(interior_value=5.0, dx=dx, side="min")
        expected_min = 5.0 - 2 * dx * 1.0  # = 4.8
        assert np.isclose(ghost_min, expected_min)

        # For max side (outward_sign = +1): ghost = interior + 2*dx*g
        ghost_max = calc.compute(interior_value=5.0, dx=dx, side="max")
        expected_max = 5.0 + 2 * dx * 1.0  # = 5.2
        assert np.isclose(ghost_max, expected_max)

    def test_robin_calculator(self):
        """Test RobinCalculator for mixed boundary conditions."""
        from mfgarchon.geometry.boundary import RobinCalculator

        # Robin: alpha*u + beta*du/dn = g
        # With alpha=1, beta=0, it reduces to Dirichlet: u = g
        calc_dirichlet = RobinCalculator(alpha=1.0, beta=0.0, rhs_value=2.0)
        ghost = calc_dirichlet.compute(interior_value=1.0, dx=0.1, side="min")
        # Should behave like Dirichlet: ghost = 2*2 - 1 = 3
        assert np.isclose(ghost, 3.0)

    def test_no_flux_calculator(self):
        """Test NoFluxCalculator (edge extension)."""
        from mfgarchon.geometry.boundary import NoFluxCalculator

        calc = NoFluxCalculator()
        ghost = calc.compute(interior_value=7.5, dx=0.1, side="max")
        assert np.isclose(ghost, 7.5)

    def test_linear_extrapolation_calculator(self):
        """Test LinearExtrapolationCalculator (zero second derivative)."""
        from mfgarchon.geometry.boundary import LinearExtrapolationCalculator

        calc = LinearExtrapolationCalculator()
        # ghost = 2*u_0 - u_1 = 2*5 - 3 = 7
        ghost = calc.compute(interior_value=5.0, dx=0.1, side="min", second_interior_value=3.0)
        assert np.isclose(ghost, 7.0)

    def test_quadratic_extrapolation_calculator(self):
        """Test QuadraticExtrapolationCalculator (zero third derivative)."""
        from mfgarchon.geometry.boundary import QuadraticExtrapolationCalculator

        calc = QuadraticExtrapolationCalculator()
        # ghost = 3*u_0 - 3*u_1 + u_2 = 3*5 - 3*3 + 1 = 7
        ghost = calc.compute(
            interior_value=5.0,
            dx=0.1,
            side="min",
            second_interior_value=3.0,
            third_interior_value=1.0,
        )
        assert np.isclose(ghost, 7.0)

    def test_fp_no_flux_calculator(self):
        """Test FPNoFluxCalculator (physics-aware zero total flux)."""
        from mfgarchon.geometry.boundary import FPNoFluxCalculator

        # Zero drift: reduces to Neumann (ghost = interior)
        calc = FPNoFluxCalculator(drift_velocity=0.0, diffusion_coeff=1.0)
        ghost = calc.compute(interior_value=3.0, dx=0.1, side="min")
        assert np.isclose(ghost, 3.0)


class TestGhostBuffer:
    """Test GhostBuffer with Topology/Calculator composition."""

    def test_ghost_buffer_periodic_2d(self):
        """Test GhostBuffer with periodic topology in 2D."""
        from mfgarchon.geometry.boundary import GhostBuffer, PeriodicTopology

        topo = PeriodicTopology(dimension=2, shape=(5, 5))
        buffer = GhostBuffer(topo)

        # Set interior to a gradient
        buffer.interior[:] = np.arange(25).reshape(5, 5)
        buffer.update()

        # Check periodic wrap-around
        # Low ghost should equal high interior
        np.testing.assert_array_equal(buffer.padded[0, 1:-1], buffer.padded[-2, 1:-1])
        # High ghost should equal low interior
        np.testing.assert_array_equal(buffer.padded[-1, 1:-1], buffer.padded[1, 1:-1])

    def test_ghost_buffer_bounded_dirichlet_2d(self):
        """Test GhostBuffer with bounded topology and Dirichlet BC."""
        from mfgarchon.geometry.boundary import (
            BoundedTopology,
            DirichletCalculator,
            GhostBuffer,
        )

        topo = BoundedTopology(dimension=2, shape=(5, 5))
        calc = DirichletCalculator(boundary_value=0.0)
        buffer = GhostBuffer(topo, calc, dx=0.1)

        buffer.interior[:] = 1.0
        buffer.update()

        # Dirichlet g=0, interior=1: ghost = 2*0 - 1 = -1
        assert np.allclose(buffer.padded[0, 1:-1], -1.0)
        assert np.allclose(buffer.padded[-1, 1:-1], -1.0)
        assert np.allclose(buffer.padded[1:-1, 0], -1.0)
        assert np.allclose(buffer.padded[1:-1, -1], -1.0)

    def test_ghost_buffer_bounded_neumann_2d(self):
        """Test GhostBuffer with bounded topology and Neumann BC (zero flux)."""
        from mfgarchon.geometry.boundary import (
            BoundedTopology,
            GhostBuffer,
            NeumannCalculator,
        )

        topo = BoundedTopology(dimension=2, shape=(5, 5))
        calc = NeumannCalculator(flux_value=0.0)
        buffer = GhostBuffer(topo, calc, dx=0.1)

        buffer.interior[:] = np.arange(25).reshape(5, 5).astype(float)
        buffer.update()

        # Zero Neumann: ghost = interior (edge extension)
        np.testing.assert_array_almost_equal(buffer.padded[0, 1:-1], buffer.interior[0, :])

    def test_ghost_buffer_bounded_requires_calculator(self):
        """Test that bounded topology requires calculator."""
        from mfgarchon.geometry.boundary import BoundedTopology, GhostBuffer

        topo = BoundedTopology(dimension=2, shape=(5, 5))
        with pytest.raises(ValueError, match="requires a BoundaryCalculator"):
            GhostBuffer(topo)  # No calculator provided

    def test_ghost_buffer_periodic_ignores_calculator(self):
        """Test that periodic topology ignores calculator (uses wrap-around)."""
        from mfgarchon.geometry.boundary import (
            DirichletCalculator,
            GhostBuffer,
            PeriodicTopology,
        )

        topo = PeriodicTopology(dimension=2, shape=(5, 5))
        calc = DirichletCalculator(boundary_value=999.0)  # Should be ignored
        buffer = GhostBuffer(topo, calc)

        buffer.interior[:] = np.arange(25).reshape(5, 5)
        buffer.update()

        # Should use wrap-around, not Dirichlet
        # If Dirichlet were used, ghost would be 2*999 - interior
        # With wrap-around, ghost[0] = interior[-1]
        np.testing.assert_array_equal(buffer.padded[0, 1:-1], buffer.padded[-2, 1:-1])

    def test_ghost_buffer_properties(self):
        """Test GhostBuffer property accessors."""
        from mfgarchon.geometry.boundary import (
            BoundedTopology,
            DirichletCalculator,
            GhostBuffer,
        )

        topo = BoundedTopology(dimension=2, shape=(10, 15))
        calc = DirichletCalculator(boundary_value=0.0)
        buffer = GhostBuffer(topo, calc, dx=(0.1, 0.2), ghost_depth=2)

        assert buffer.shape == (10, 15)
        assert buffer.padded_shape == (14, 19)
        assert buffer.ghost_depth == 2
        assert buffer.dx == (0.1, 0.2)
        assert buffer.topology is topo
        assert buffer.calculator is calc

    def test_ghost_buffer_reset(self):
        """Test GhostBuffer reset method."""
        from mfgarchon.geometry.boundary import GhostBuffer, PeriodicTopology

        topo = PeriodicTopology(dimension=1, shape=(10,))
        buffer = GhostBuffer(topo)

        buffer.interior[:] = 5.0
        buffer.reset(fill_value=0.0)

        assert np.allclose(buffer.padded, 0.0)

    def test_ghost_buffer_copy_to_interior(self):
        """Test GhostBuffer copy_to_interior method."""
        from mfgarchon.geometry.boundary import GhostBuffer, PeriodicTopology

        topo = PeriodicTopology(dimension=2, shape=(3, 3))
        buffer = GhostBuffer(topo)

        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        buffer.copy_to_interior(data)

        np.testing.assert_array_equal(buffer.interior, data)

    def test_ghost_buffer_3d_periodic(self):
        """Test GhostBuffer with 3D periodic topology."""
        from mfgarchon.geometry.boundary import GhostBuffer, PeriodicTopology

        topo = PeriodicTopology(dimension=3, shape=(4, 4, 4))
        buffer = GhostBuffer(topo)

        # Set a 3D gradient
        for i in range(4):
            buffer.interior[i, :, :] = float(i)

        buffer.update()

        # Check wrap-around on first axis
        np.testing.assert_array_equal(buffer.padded[0, 1:-1, 1:-1], buffer.padded[-2, 1:-1, 1:-1])
        np.testing.assert_array_equal(buffer.padded[-1, 1:-1, 1:-1], buffer.padded[1, 1:-1, 1:-1])


class TestLinearReflectionNeumannFlux:
    """Nonzero Neumann flux in the order<=2 linear-reflection ghost path (Issue #1186 sibling).

    The FDM/SL ghost path (``_apply_linear_reflection``) used a pure mirror for
    Neumann/no-flux, silently dropping a nonzero ``neumann_bc(value=g)`` (it always encoded
    du/dn=0). The fix adds the linear flux offset (Robin-branch sign convention). g=0 stays a
    pure mirror -> byte-identical for the no-flux/zero-Neumann case the paper uses (EOC-safe).
    """

    @staticmethod
    def _ghosts(bc, u, dx=0.1, g=1):
        from mfgarchon.geometry.boundary.applicator_fdm import PreallocatedGhostBuffer

        buf = PreallocatedGhostBuffer(
            interior_shape=(len(u),),
            boundary_conditions=bc,
            domain_bounds=np.array([[0.0, (len(u) - 1) * dx]]),
            order=2,  # order<=2 -> linear reflection path
            ghost_depth=g,
        )
        buf.interior[:] = u
        buf.update_ghosts()
        return buf.padded.copy()

    def test_zero_neumann_byte_identical_to_no_flux(self):
        """g=0 must reproduce the pure mirror (no-flux), bit-for-bit -- the EOC-safe property."""
        u = np.sin(np.linspace(0.0, 1.0, 11))
        g_noflux = self._ghosts(no_flux_bc(dimension=1), u)
        g_neumann0 = self._ghosts(neumann_bc(dimension=1, value=0.0), u)
        np.testing.assert_array_equal(g_neumann0, g_noflux)

    def test_nonzero_neumann_flux_recovered(self):
        """A nonzero neumann_bc(value=v) is applied (not dropped): the cell-centered ghost
        encodes du/dn = v at both walls (was silently 0)."""
        dx = 0.1
        u = np.sin(np.linspace(0.0, 1.0, 11))
        for v in (0.5, -1.3):
            gh = self._ghosts(neumann_bc(dimension=1, value=v), u, dx=dx)
            # du/dn is the OUTWARD-normal derivative: low wall outward = -x, so
            # du/dn = (u_ghost - u_interior)/dx (Issue #1262, 2026-06-10 audit). The previous
            # (u_interior - u_ghost)/dx computed -du/dn; that sign error cancelled the old
            # ghost-sign bug (u_g = u_i - dx*v), so this test passed against incorrect ghosts.
            # With the corrected ghost (u_g = u_i + dx*v) the low-wall du/dn is +v.
            dudn_low = (gh[0] - gh[1]) / dx  # (u_ghost - u_interior)/dx, low outward = -x
            dudn_high = (gh[-1] - gh[-2]) / dx  # (u_ghost - u_interior)/dx, high outward = +x
            assert abs(dudn_low - v) < 1e-12, f"low du/dn={dudn_low} != {v}"
            assert abs(dudn_high - v) < 1e-12, f"high du/dn={dudn_high} != {v}"

    def test_no_flux_ignores_stray_value(self):
        """NO_FLUX is definitionally zero-flux: the mirror is unchanged regardless of input."""
        u = np.cos(np.linspace(0.0, 1.0, 11))
        # no_flux must equal a true zero-Neumann mirror (no offset applied to NO_FLUX).
        np.testing.assert_array_equal(
            self._ghosts(no_flux_bc(dimension=1), u), self._ghosts(neumann_bc(dimension=1, value=0.0), u)
        )
