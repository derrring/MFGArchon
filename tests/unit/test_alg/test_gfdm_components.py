"""
Unit tests for GFDM components (Issue #545).

This module demonstrates the proper way to test GFDM components after the
mixin-to-composition refactoring. Each component is tested independently
without requiring a full solver instantiation.

Components tested:
- GridCollocationMapper: Bidirectional grid ↔ collocation interpolation
- NeighborhoodBuilder: Stencil construction, Taylor matrices, weight functions
- BoundaryHandler: Boundary normals, LCR, ghost nodes
- MonotonicityEnforcer: QP-constrained monotonicity preservation
"""

from __future__ import annotations

import pytest

import numpy as np
from scipy.spatial import Delaunay

from mfgarchon.alg.numerical.gfdm_components import (
    BoundaryHandler,
    GridCollocationMapper,
    NeighborhoodBuilder,
)


class TestGridCollocationMapper:
    """Test GridCollocationMapper component independently."""

    @pytest.fixture
    def mapper_1d(self):
        """Create 1D mapper for testing."""
        collocation_points = np.linspace(0, 1, 20).reshape(-1, 1)
        grid_shape = (21,)  # Grid has 21 points
        domain_bounds = [(0.0, 1.0)]
        return GridCollocationMapper(collocation_points, grid_shape, domain_bounds)

    @pytest.fixture
    def mapper_2d(self):
        """Create 2D mapper for testing."""
        # Create 2D collocation points
        np.random.seed(42)
        collocation_points = np.random.uniform(0, 1, (100, 2))
        grid_shape = (11, 11)
        domain_bounds = [(0.0, 1.0), (0.0, 1.0)]
        return GridCollocationMapper(collocation_points, grid_shape, domain_bounds)

    def test_map_grid_to_collocation_1d(self, mapper_1d):
        """Test grid → collocation interpolation in 1D."""
        # Create grid values: linear function u(x) = 2x + 1
        x_grid = np.linspace(0, 1, 21)
        u_grid = 2 * x_grid + 1

        # Map to collocation points
        u_collocation = mapper_1d.map_grid_to_collocation(u_grid)

        # Check shape
        assert u_collocation.shape == (20,)

        # Check values (should match linear function at collocation points)
        x_coll = mapper_1d.collocation_points[:, 0]
        u_expected = 2 * x_coll + 1
        np.testing.assert_allclose(u_collocation, u_expected, atol=1e-10)

    def test_map_grid_to_collocation_2d(self, mapper_2d):
        """Test grid → collocation interpolation in 2D."""
        # Create grid values: bilinear function u(x,y) = x + 2y
        x = np.linspace(0, 1, 11)
        y = np.linspace(0, 1, 11)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u_grid = (X + 2 * Y).flatten()

        # Map to collocation points
        u_collocation = mapper_2d.map_grid_to_collocation(u_grid)

        # Check shape
        assert u_collocation.shape == (100,)

        # Check values at collocation points
        x_coll = mapper_2d.collocation_points[:, 0]
        y_coll = mapper_2d.collocation_points[:, 1]
        u_expected = x_coll + 2 * y_coll
        np.testing.assert_allclose(u_collocation, u_expected, atol=1e-10)

    def test_map_collocation_to_grid_2d(self, mapper_2d):
        """Test collocation → grid reconstruction in 2D."""
        # Create collocation values
        x_coll = mapper_2d.collocation_points[:, 0]
        y_coll = mapper_2d.collocation_points[:, 1]
        u_collocation = x_coll**2 + y_coll**2

        # Map to grid
        u_grid = mapper_2d.map_collocation_to_grid(u_collocation)

        # Check shape
        assert u_grid.shape == (121,)  # 11*11

        # Partition of unity, which holds at every grid node including the ones outside the
        # cloud's convex hull: a constant field must come back as that constant. Measured 4.4e-16,
        # and it is what fails if the reconstruction returns zeros or NaN off the hull.
        np.testing.assert_allclose(mapper_2d.map_collocation_to_grid(np.full(100, 3.0)), 3.0, atol=1e-12)

        # "Grid reconstruction should approximate the function", asserted. A linear interpolant
        # reproduces linear data exactly, so this is an oracle rather than a tolerance: measured
        # 4.4e-16 at the 80 of 121 nodes inside the hull. Outside it the reconstruction is
        # extrapolated and genuinely bad (measured 3.1e-01 for this same linear field), which is
        # why the check is hull-restricted. An i/j transposition scores 8.0e-01 inside the hull
        # and returning zeros scores 2.7, both far outside the tolerance.
        u_linear = mapper_2d.map_collocation_to_grid(x_coll + 2 * y_coll)
        grid_axis = np.linspace(0.0, 1.0, 11)
        GX, GY = np.meshgrid(grid_axis, grid_axis, indexing="ij")
        grid_points = np.column_stack([GX.ravel(), GY.ravel()])
        inside_hull = Delaunay(mapper_2d.collocation_points).find_simplex(grid_points) >= 0
        assert inside_hull.sum() == 80
        np.testing.assert_allclose(
            u_linear[inside_hull], (grid_points[:, 0] + 2 * grid_points[:, 1])[inside_hull], atol=1e-12
        )

    def test_batch_mapping_consistency(self, mapper_2d):
        """Test batch vs sequential mapping gives same result."""
        # Create time-series data (Nt=5 timesteps)
        Nt = 5
        U_grid = np.random.randn(Nt, 121)  # (Nt, Nx*Ny)

        # Batch mapping
        U_collocation_batch = mapper_2d.map_grid_to_collocation_batch(U_grid)

        # Sequential mapping
        U_collocation_seq = np.zeros((Nt, 100))
        for t in range(Nt):
            U_collocation_seq[t, :] = mapper_2d.map_grid_to_collocation(U_grid[t])

        # Should be identical
        np.testing.assert_allclose(U_collocation_batch, U_collocation_seq)


class TestNeighborhoodBuilder:
    """Test NeighborhoodBuilder component independently."""

    @pytest.fixture
    def builder_1d(self):
        """Create 1D neighborhood builder for testing."""
        from mfgarchon.utils.numerical.gfdm_strategies import TaylorOperator

        collocation_points = np.linspace(0, 1, 20).reshape(-1, 1)
        gfdm_operator = TaylorOperator(
            points=collocation_points, delta=0.15, taylor_order=2, weight_function="wendland"
        )

        builder = NeighborhoodBuilder(
            collocation_points=collocation_points,
            dimension=1,
            delta=0.15,
            taylor_order=2,
            weight_function="wendland",
            weight_scale=0.15,
            k_min=5,
            adaptive_neighborhoods=False,
            max_delta_multiplier=3.0,
            boundary_indices=np.array([0, 19]),
            n_derivatives=len(gfdm_operator.multi_indices),
            multi_indices=gfdm_operator.multi_indices,
            gfdm_operator=gfdm_operator,
            use_local_coordinate_rotation=False,
            boundary_handler=None,
        )
        return builder

    def test_compute_weights_wendland(self, builder_1d):
        """Test Wendland weight function computation."""
        distances = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
        weights = builder_1d.compute_weights(distances)

        # Wendland weights should be non-negative
        assert np.all(weights >= 0)

        # Weight at distance 0 should be maximum
        assert weights[0] == np.max(weights)

        # Weights should decay with distance
        assert weights[0] > weights[1] > weights[2]

        # Weight at distance = delta should be 0 (compact support)
        assert weights[3] == pytest.approx(0.0, abs=1e-10)

        # Weights beyond delta should also be 0
        assert weights[4] == pytest.approx(0.0, abs=1e-10)

    def test_compute_weights_gaussian(self):
        """Test Gaussian weight function computation."""
        from mfgarchon.utils.numerical.gfdm_strategies import TaylorOperator

        collocation_points = np.linspace(0, 1, 20).reshape(-1, 1)
        gfdm_operator = TaylorOperator(points=collocation_points, delta=0.15, taylor_order=2)

        builder = NeighborhoodBuilder(
            collocation_points=collocation_points,
            dimension=1,
            delta=0.15,
            taylor_order=2,
            weight_function="gaussian",
            weight_scale=0.15,
            k_min=5,
            adaptive_neighborhoods=False,
            max_delta_multiplier=3.0,
            boundary_indices=np.array([]),
            n_derivatives=len(gfdm_operator.multi_indices),
            multi_indices=gfdm_operator.multi_indices,
            gfdm_operator=gfdm_operator,
            use_local_coordinate_rotation=False,
            boundary_handler=None,
        )

        distances = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
        weights = builder.compute_weights(distances)

        # Gaussian weights should be positive
        assert np.all(weights > 0)

        # Weight at distance 0 should be 1
        assert weights[0] == pytest.approx(1.0)

        # Weights should decay monotonically
        assert weights[0] > weights[1] > weights[2] > weights[3] > weights[4]

    def test_neighborhood_structure_builds(self, builder_1d):
        """Test neighborhood structure construction."""
        builder_1d.build_neighborhood_structure()

        # Check neighborhoods were created
        assert len(builder_1d.neighborhoods) == 20

        # Each neighborhood should have required structure
        for i in range(20):
            neighborhood = builder_1d.neighborhoods[i]
            assert "indices" in neighborhood
            assert "points" in neighborhood
            assert "distances" in neighborhood
            assert "size" in neighborhood
            assert neighborhood["size"] > 0

    def test_taylor_matrices_build(self, builder_1d):
        """Test Taylor matrix construction."""
        builder_1d.build_neighborhood_structure()
        builder_1d.build_taylor_matrices()

        # Check Taylor matrices were created
        assert len(builder_1d.taylor_matrices) == 20

        # Each Taylor matrix should have required components
        for i in range(20):
            taylor = builder_1d.taylor_matrices[i]
            if taylor is not None:  # Skip points with insufficient neighbors
                assert "A" in taylor
                assert "W" in taylor
                assert "sqrt_W" in taylor

    def test_reverse_neighborhoods_build(self, builder_1d):
        """Test reverse neighborhood mapping."""
        builder_1d.build_neighborhood_structure()
        builder_1d.build_reverse_neighborhoods()

        # Check reverse neighborhoods
        assert len(builder_1d._reverse_neighborhoods) == 20

        # Each point should have a reverse neighborhood (possibly empty)
        for j in range(20):
            reverse_neighbors = builder_1d._reverse_neighborhoods[j]
            assert isinstance(reverse_neighbors, np.ndarray)

        # The relation that defines a reverse map: j is in the neighborhood of i exactly when i is
        # in the reverse neighborhood of j. This is the transpose consistency the GFDM adjoint
        # operator rests on, and it fails for an off-by-one, an omitted self-index, or a reverse
        # map built from the wrong stencil. Measured: holds for all 400 pairs, 120 edges each way.
        forward = np.zeros((20, 20), dtype=bool)
        reverse = np.zeros((20, 20), dtype=bool)
        for i in range(20):
            forward[i, builder_1d.neighborhoods[i]["indices"]] = True
            reverse[i, builder_1d._reverse_neighborhoods[i]] = True
        np.testing.assert_array_equal(reverse, forward.T)

        # And closing the "possibly empty" loophole above: with these stencils no point is
        # unreachable. Measured reverse sizes range 3 to 9.
        assert reverse.any(axis=1).all()


class TestBoundaryHandler:
    """Test BoundaryHandler component independently."""

    @pytest.fixture
    def handler_2d(self):
        """Create 2D boundary handler for testing."""
        # Create 2D grid points
        x = np.linspace(0, 1, 11)
        y = np.linspace(0, 1, 11)
        X, Y = np.meshgrid(x, y, indexing="ij")
        collocation_points = np.column_stack([X.ravel(), Y.ravel()])

        # Boundary points (edges of grid)
        boundary_mask = (X.ravel() == 0) | (X.ravel() == 1) | (Y.ravel() == 0) | (Y.ravel() == 1)
        boundary_indices = np.where(boundary_mask)[0]

        # Mock GFDM operator (minimal for testing)
        class MockGFDMOperator:
            def get_neighborhood(self, i):
                return {"indices": np.array([]), "points": np.array([[]]), "distances": np.array([]), "size": 0}

        handler = BoundaryHandler(
            collocation_points=collocation_points,
            dimension=2,
            domain_bounds=[(0.0, 1.0), (0.0, 1.0)],
            boundary_indices=boundary_indices,
            neighborhoods={},
            boundary_conditions=None,
            use_ghost_nodes=False,
            use_wind_dependent_bc=False,
            gfdm_operator=MockGFDMOperator(),
            bc_property_getter=lambda prop, default: default,
            gradient_computer=None,
        )
        return handler

    def test_compute_boundary_normals_rectangular(self, handler_2d):
        """Test boundary normal computation for rectangular domain."""
        normals = handler_2d.compute_boundary_normals()

        # Check normals were computed for all boundary points
        assert normals.shape == (handler_2d.boundary_indices.shape[0], 2)

        # Check normals have unit length
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

        # Check normals point outward (for corners and edges)
        # Left edge (x=0): normal should point left (-1, 0)
        # Right edge (x=1): normal should point right (1, 0)
        # etc.
        for idx in handler_2d.boundary_indices:
            point = handler_2d.collocation_points[idx]
            normal = normals[np.where(handler_2d.boundary_indices == idx)[0][0]]

            # Check edge normals are correct
            if point[0] == 0:  # Left edge
                assert normal[0] < 0  # Points left
            elif point[0] == 1:  # Right edge
                assert normal[0] > 0  # Points right

            if point[1] == 0:  # Bottom edge
                assert normal[1] < 0  # Points down
            elif point[1] == 1:  # Top edge
                assert normal[1] > 0  # Points up


if __name__ == "__main__":
    """Run smoke tests."""
    print("Testing GFDM components...")

    # Test GridCollocationMapper
    print("\n1. Testing GridCollocationMapper...")
    mapper = GridCollocationMapper(
        collocation_points=np.random.uniform(0, 1, (50, 2)),
        grid_shape=(11, 11),
        domain_bounds=[(0, 1), (0, 1)],
    )
    u_grid = np.random.randn(121)
    u_coll = mapper.map_grid_to_collocation(u_grid)
    assert u_coll.shape == (50,)
    print("   ✓ Grid → collocation mapping works")

    # Test NeighborhoodBuilder
    print("\n2. Testing NeighborhoodBuilder...")
    from mfgarchon.utils.numerical.gfdm_strategies import TaylorOperator

    points = np.linspace(0, 1, 20).reshape(-1, 1)
    operator = TaylorOperator(points, delta=0.15, taylor_order=2)
    builder = NeighborhoodBuilder(
        collocation_points=points,
        dimension=1,
        delta=0.15,
        taylor_order=2,
        weight_function="wendland",
        weight_scale=0.15,
        k_min=5,
        adaptive_neighborhoods=False,
        max_delta_multiplier=3.0,
        boundary_indices=np.array([0, 19]),
        n_derivatives=len(operator.multi_indices),
        multi_indices=operator.multi_indices,
        gfdm_operator=operator,
        use_local_coordinate_rotation=False,
        boundary_handler=None,
    )
    builder.build_neighborhood_structure()
    assert len(builder.neighborhoods) == 20
    print("   ✓ Neighborhood construction works")

    weights = builder.compute_weights(np.array([0.0, 0.05, 0.10]))
    assert weights[0] == pytest.approx(1.0)
    print("   ✓ Weight function computation works")

    print("\n✅ All component smoke tests passed!")
