"""
Tests for particle-to-particle geometry projection (Issue #269).

Tests the PointCloudGeometry class and GeometryProjector with particle-particle
projection patterns:
1. Fixed collocation + moving particles: Requires RBF/KDE projection
2. Co-moving particles: Identity projection optimization

Created: 2025-11-12
"""

import numpy as np

from mfgarchon.geometry import GeometryProjector, PointCloudGeometry


class TestPointCloudGeometry:
    """Tests for PointCloudGeometry wrapper class."""

    def test_init_1d(self):
        """Test 1D point cloud initialization."""
        positions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        geom = PointCloudGeometry(positions)

        assert geom.dimension == 1
        assert geom.num_particles == 5
        assert geom.num_spatial_points == 5
        assert geom.positions.shape == (5, 1)

    def test_init_2d(self):
        """Test 2D point cloud initialization."""
        positions = np.random.uniform(0, 1, (100, 2))
        geom = PointCloudGeometry(positions)

        assert geom.dimension == 2
        assert geom.num_particles == 100
        assert geom.positions.shape == (100, 2)

    def test_init_3d(self):
        """Test 3D point cloud initialization."""
        positions = np.random.uniform(0, 1, (50, 3))
        geom = PointCloudGeometry(positions)

        assert geom.dimension == 3
        assert geom.num_particles == 50

    def test_bounds(self):
        """Test bounding box computation."""
        positions = np.array([[0.1, 0.2], [0.9, 0.8], [0.5, 0.5]])
        geom = PointCloudGeometry(positions)

        min_coords, max_coords = geom.bounds
        np.testing.assert_array_almost_equal(min_coords, [0.1, 0.2])
        np.testing.assert_array_almost_equal(max_coords, [0.9, 0.8])

    def test_is_same_pointset_identical(self):
        """Test identity detection for same array reference."""
        positions = np.random.uniform(0, 1, (100, 2))
        geom1 = PointCloudGeometry(positions)
        geom2 = PointCloudGeometry(positions)  # Same array

        assert geom1.is_same_pointset(geom2)

    def test_is_same_pointset_equal_values(self):
        """Test identity detection for equal but different arrays."""
        positions1 = np.random.uniform(0, 1, (100, 2))
        positions2 = positions1.copy()

        geom1 = PointCloudGeometry(positions1)
        geom2 = PointCloudGeometry(positions2)

        assert geom1.is_same_pointset(geom2)

    def test_is_same_pointset_different(self):
        """Test that different point sets are detected."""
        positions1 = np.random.uniform(0, 1, (100, 2))
        positions2 = np.random.uniform(0, 1, (100, 2))

        geom1 = PointCloudGeometry(positions1)
        geom2 = PointCloudGeometry(positions2)

        assert not geom1.is_same_pointset(geom2)

    def test_is_same_pointset_different_sizes(self):
        """Test that point sets with different sizes are not equal."""
        positions1 = np.random.uniform(0, 1, (100, 2))
        positions2 = np.random.uniform(0, 1, (50, 2))

        geom1 = PointCloudGeometry(positions1)
        geom2 = PointCloudGeometry(positions2)

        assert not geom1.is_same_pointset(geom2)

    def test_get_problem_config(self):
        """Test problem configuration generation."""
        positions = np.array([[0.1, 0.2], [0.9, 0.8], [0.5, 0.5]])
        geom = PointCloudGeometry(positions)

        config = geom.get_problem_config()

        assert config["num_spatial_points"] == 3
        assert config["spatial_shape"] == (3,)
        assert len(config["spatial_bounds"]) == 2  # 2D
        assert config["spatial_discretization"] is None


class TestParticleParticleProjection:
    """Tests for particle-to-particle geometry projection."""

    def test_co_moving_particles_identity_hjb_to_fp(self):
        """Test identity projection for co-moving particles (HJB → FP)."""
        # Same particle set for both HJB and FP
        particles = np.random.uniform(0, 1, (100, 2))
        hjb_geom = PointCloudGeometry(particles)
        fp_geom = PointCloudGeometry(particles)

        projector = GeometryProjector(hjb_geom, fp_geom)

        # Should detect identity case
        assert projector.hjb_to_fp_method == "identity"

        # Value function projection should return input unchanged
        U = np.random.rand(100)
        U_projected = projector.project_hjb_to_fp(U)

        np.testing.assert_array_equal(U_projected, U)

    def test_co_moving_particles_identity_fp_to_hjb(self):
        """Test identity projection for co-moving particles (FP → HJB)."""
        # Same particle set for both HJB and FP
        particles = np.random.uniform(0, 1, (100, 2))
        hjb_geom = PointCloudGeometry(particles)
        fp_geom = PointCloudGeometry(particles)

        projector = GeometryProjector(hjb_geom, fp_geom)

        # Should detect identity case
        assert projector.fp_to_hjb_method == "identity"

        # Density projection should return input unchanged
        M = np.random.rand(100)
        M_projected = projector.project_fp_to_hjb(M)

        np.testing.assert_array_equal(M_projected, M)

    def test_different_particles_rbf_detection(self):
        """Test RBF detection for different particle sets."""
        hjb_particles = np.random.uniform(0, 1, (50, 2))
        fp_particles = np.random.uniform(0, 1, (100, 2))

        hjb_geom = PointCloudGeometry(hjb_particles)
        fp_geom = PointCloudGeometry(fp_particles)

        projector = GeometryProjector(hjb_geom, fp_geom)

        # Should detect particle_rbf for HJB → FP
        assert projector.hjb_to_fp_method == "particle_rbf"

        # Should detect particle_kde for FP → HJB
        assert projector.fp_to_hjb_method == "particle_kde"

    def test_rbf_interpolation_1d(self):
        """Test RBF interpolation for 1D value function."""
        # Source particles (collocation points)
        source_x = np.linspace(0, 1, 20)
        # Target particles (FP particles)
        target_x = np.random.uniform(0, 1, 50)

        hjb_geom = PointCloudGeometry(source_x)
        fp_geom = PointCloudGeometry(target_x)

        projector = GeometryProjector(hjb_geom, fp_geom)

        # Test function: U(x) = sin(2πx)
        U_source = np.sin(2 * np.pi * source_x)

        # Project to target
        U_target = projector.project_hjb_to_fp(U_source)

        # Check shape
        assert U_target.shape == (50,)

        # Check interpolation accuracy at a few points
        # RBF should be accurate for smooth functions
        U_expected = np.sin(2 * np.pi * target_x)
        np.testing.assert_allclose(U_target, U_expected, rtol=0.1, atol=0.05)

    def test_rbf_interpolation_2d(self):
        """Test RBF interpolation for 2D value function."""
        # Source particles (collocation points) - structured grid
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y)
        source_positions = np.column_stack([X.ravel(), Y.ravel()])

        # Target particles (FP particles) - random
        target_positions = np.random.uniform(0, 1, (50, 2))

        hjb_geom = PointCloudGeometry(source_positions)
        fp_geom = PointCloudGeometry(target_positions)

        projector = GeometryProjector(hjb_geom, fp_geom)

        # Test function: U(x,y) = exp(-10*((x-0.5)^2 + (y-0.5)^2))
        U_source = np.exp(-10 * ((source_positions[:, 0] - 0.5) ** 2 + (source_positions[:, 1] - 0.5) ** 2))

        # Project to target
        U_target = projector.project_hjb_to_fp(U_source)

        # Check shape
        assert U_target.shape == (50,)

        # Check interpolation accuracy
        U_expected = np.exp(-10 * ((target_positions[:, 0] - 0.5) ** 2 + (target_positions[:, 1] - 0.5) ** 2))
        np.testing.assert_allclose(U_target, U_expected, rtol=0.15, atol=0.05)

    def test_kde_projection_1d(self):
        """Test KDE projection for 1D density."""
        # Source particles (FP) with uniform density
        source_x = np.random.uniform(0, 1, 200)
        # Target particles (HJB collocation)
        target_x = np.linspace(0, 1, 50)

        fp_geom = PointCloudGeometry(source_x)
        hjb_geom = PointCloudGeometry(target_x)

        projector = GeometryProjector(hjb_geom, fp_geom)

        # Uniform weights
        M_source = np.ones(200) / 200

        # Project to target
        M_target = projector.project_fp_to_hjb(M_source, bandwidth="scott")

        # Check shape
        assert M_target.shape == (50,)

        # Check that density is positive
        assert np.all(M_target > 0)

        # Check approximate normalization (KDE should preserve total mass)
        # Note: This is approximate due to bandwidth effects
        assert np.sum(M_target) > 0.5  # Loose check

    def test_kde_projection_2d(self):
        """Test KDE projection for 2D density."""
        # Source particles (FP) clustered around (0.5, 0.5)
        source_positions = 0.5 + 0.1 * np.random.randn(200, 2)
        source_positions = np.clip(source_positions, 0, 1)

        # Target particles (HJB collocation) - regular grid
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y)
        target_positions = np.column_stack([X.ravel(), Y.ravel()])

        fp_geom = PointCloudGeometry(source_positions)
        hjb_geom = PointCloudGeometry(target_positions)

        projector = GeometryProjector(hjb_geom, fp_geom)

        # Uniform weights
        M_source = np.ones(200) / 200

        # Project to target
        M_target = projector.project_fp_to_hjb(M_source, bandwidth="scott")

        # Check shape
        assert M_target.shape == (100,)

        # Check that density is positive
        assert np.all(M_target > 0)

        # Density should be highest near (0.5, 0.5)
        center_idx = 55  # Approximately (0.5, 0.5) in flattened grid
        corner_idx = 0  # (0, 0)
        assert M_target[center_idx] > M_target[corner_idx]

    def test_3d_particle_projection(self):
        """Test particle projection in 3D."""
        hjb_particles = np.random.uniform(0, 1, (50, 3))
        fp_particles = np.random.uniform(0, 1, (100, 3))

        hjb_geom = PointCloudGeometry(hjb_particles)
        fp_geom = PointCloudGeometry(fp_particles)

        projector = GeometryProjector(hjb_geom, fp_geom)

        # Test both directions
        U_hjb = np.random.rand(50)
        U_fp = projector.project_hjb_to_fp(U_hjb)
        assert U_fp.shape == (100,)

        M_fp = np.random.rand(100)
        M_hjb = projector.project_fp_to_hjb(M_fp)
        assert M_hjb.shape == (50,)

    def test_metadata_preservation(self):
        """Test that metadata is preserved in PointCloudGeometry."""
        positions = np.random.uniform(0, 1, (100, 2))
        metadata = {"particle_ids": np.arange(100), "weights": np.ones(100)}

        geom = PointCloudGeometry(positions, metadata=metadata)

        assert "particle_ids" in geom.metadata
        assert "weights" in geom.metadata
        assert len(geom.metadata["particle_ids"]) == 100


if __name__ == "__main__":
    # Quick smoke test
    print("Running particle-particle projection smoke tests...")

    # Test 1: Co-moving particles (identity)
    particles = np.random.uniform(0, 1, (100, 2))
    hjb_geom = PointCloudGeometry(particles)
    fp_geom = PointCloudGeometry(particles)
    projector = GeometryProjector(hjb_geom, fp_geom)

    assert projector.hjb_to_fp_method == "identity"
    assert projector.fp_to_hjb_method == "identity"
    print("✓ Co-moving particles (identity projection)")

    # Test 2: Different particles (RBF + KDE)
    hjb_particles = np.random.uniform(0, 1, (50, 2))
    fp_particles = np.random.uniform(0, 1, (100, 2))
    hjb_geom = PointCloudGeometry(hjb_particles)
    fp_geom = PointCloudGeometry(fp_particles)
    projector = GeometryProjector(hjb_geom, fp_geom)

    assert projector.hjb_to_fp_method == "particle_rbf"
    assert projector.fp_to_hjb_method == "particle_kde"
    print("✓ Different particles (RBF + KDE projection)")

    # Test 3: RBF interpolation
    U_hjb = np.random.rand(50)
    U_fp = projector.project_hjb_to_fp(U_hjb)
    assert U_fp.shape == (100,)
    print("✓ RBF interpolation (50 → 100 particles)")

    # Test 4: KDE projection
    M_fp = np.ones(100) / 100
    M_hjb = projector.project_fp_to_hjb(M_fp, bandwidth="scott")
    assert M_hjb.shape == (50,)
    assert np.all(M_hjb > 0)
    print("✓ KDE projection (100 → 50 particles)")

    print("\nAll smoke tests passed! ✓")
