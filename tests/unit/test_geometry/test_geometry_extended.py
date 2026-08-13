"""
Extended tests for implicit domain geometry: edge cases and high dimensions.

Run with: python -m pytest tests/unit/geometry/test_geometry_extended.py -v
"""

import pytest

import numpy as np

from mfgarchon.geometry.implicit import (
    ComplementDomain,
    DifferenceDomain,
    Hyperrectangle,
    Hypersphere,
    IntersectionDomain,
    UnionDomain,
)


class TestHighDimensionalDomains:
    """Test geometry infrastructure in high dimensions (d>=5)."""

    def test_5d_hypercube(self):
        """Test 5D unit hypercube."""
        domain = Hyperrectangle(np.array([[0, 1]] * 5))

        assert domain.dimension == 5
        particles = domain.sample_uniform(500, seed=42)
        assert particles.shape == (500, 5)
        assert np.all(domain.contains(particles))
        assert np.all(particles >= 0)
        assert np.all(particles <= 1)

    def test_6d_hypersphere(self):
        """Test 6D hypersphere."""
        domain = Hypersphere(center=[0.5] * 6, radius=0.3)

        assert domain.dimension == 6
        particles = domain.sample_uniform(500, seed=42)
        assert particles.shape == (500, 6)
        assert np.all(domain.contains(particles))

        # Verify particles are within radius
        distances = np.linalg.norm(particles - 0.5, axis=1)
        assert np.all(distances <= 0.3)

    def test_10d_domain_with_obstacle(self):
        """Test 10D domain with spherical obstacle."""
        base = Hyperrectangle(np.array([[-1, 1]] * 10))
        obstacle = Hypersphere(center=[0] * 10, radius=0.3)
        domain = DifferenceDomain(base, obstacle)

        assert domain.dimension == 10
        particles = domain.sample_uniform(200, seed=42)
        assert particles.shape == (200, 10)

        # Verify particles avoid obstacle
        distances = np.linalg.norm(particles, axis=1)
        assert np.all(distances > 0.3)

    def test_volume_scaling_high_dim(self):
        """Test volume computation scales correctly in high dimensions."""
        # Unit hypercube: volume = 1 for all dimensions
        for d in [2, 3, 4, 5, 6]:
            domain = Hyperrectangle(np.array([[0, 1]] * d))
            volume = domain.compute_volume()
            assert np.abs(volume - 1.0) < 1e-10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_hyperrectangle_boundary_points(self):
        """Test signed distance at boundary points."""
        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))

        # Corners
        assert np.abs(domain.signed_distance(np.array([0, 0]))) < 1e-10
        assert np.abs(domain.signed_distance(np.array([1, 1]))) < 1e-10

        # Edges
        assert np.abs(domain.signed_distance(np.array([0.5, 0]))) < 1e-10
        assert np.abs(domain.signed_distance(np.array([1, 0.5]))) < 1e-10

    def test_hypersphere_boundary_points(self):
        """Test signed distance at boundary of sphere."""
        sphere = Hypersphere(center=[0, 0], radius=1.0)

        # Points on boundary (distance = radius)
        boundary_points = np.array(
            [
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [np.sqrt(0.5), np.sqrt(0.5)],
            ]
        )

        distances = sphere.signed_distance(boundary_points)
        assert np.all(np.abs(distances) < 1e-10)

    def test_empty_intersection(self):
        """Test intersection of non-overlapping domains."""
        rect1 = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        rect2 = Hyperrectangle(np.array([[2, 3], [2, 3]]))
        intersection = IntersectionDomain([rect1, rect2])

        # Should have no interior points
        test_points = np.array([[0.5, 0.5], [2.5, 2.5], [1.5, 1.5]])
        assert not np.any(intersection.contains(test_points))

    def test_thin_domain(self):
        """Test domain with very different aspect ratios."""
        thin_rect = Hyperrectangle(np.array([[0, 10], [0, 0.01]]))

        particles = thin_rect.sample_uniform(500, seed=42)
        assert particles.shape == (500, 2)
        assert np.all(particles[:, 0] >= 0)
        assert np.all(particles[:, 0] <= 10)
        assert np.all(particles[:, 1] >= 0)
        assert np.all(particles[:, 1] <= 0.01)


class TestBoundaryConditions:
    """Test boundary condition handling."""

    def test_reflecting_bc_hyperrectangle(self):
        """Reflecting BC must reflect, not merely land the particle inside the box.

        ``contains()`` cannot tell a reflection from a clamp: projecting 1.2 to the wall at 1.0
        satisfies it exactly as well as reflecting it to 0.8, and measured, ``contains()``
        returns True on all four points of the clamped answer. So pin the reflected position,
        ``2*bound - x``, which is what the operation means.

        Measured deviation from the exact reflections below: 8.3e-17, so atol=1e-12 is four
        orders above the floating-point floor. A clamp lands 0.3 away -- eleven orders above it.
        """
        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))

        # Particles outside domain
        particles_outside = np.array(
            [
                [1.2, 0.5],
                [-0.1, 0.5],
                [0.5, 1.3],
                [0.5, -0.2],
            ]
        )

        particles_reflected = domain.apply_boundary_conditions(particles_outside, bc_type="reflecting")

        np.testing.assert_allclose(
            particles_reflected,
            np.array([[0.8, 0.5], [0.1, 0.5], [0.5, 0.7], [0.5, 0.2]]),
            atol=1e-12,
            err_msg="reflecting BC must map x to 2*bound - x, not clamp it to the wall",
        )

    def test_absorbing_bc(self):
        """Test absorbing boundary conditions."""
        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))

        # Mix of inside and outside particles
        particles = np.array(
            [
                [0.5, 0.5],  # Inside
                [1.5, 0.5],  # Outside
                [0.3, 0.3],  # Inside
                [-0.1, 0.5],  # Outside
            ]
        )

        particles_absorbed = domain.apply_boundary_conditions(particles, bc_type="absorbing")

        # The survivor count alone does not say WHICH particles survived, nor that input order is
        # kept -- "keep the first two rows" and an inverted mask both give a (2, 2) array whose
        # rows are inside. Measured, the surviving rows are copied verbatim, so pin them exactly.
        np.testing.assert_array_equal(particles_absorbed, np.array([[0.5, 0.5], [0.3, 0.3]]))


class TestCSGComplexity:
    """Test complex CSG operations."""

    def test_union_of_many_domains(self):
        """Test union of multiple domains."""
        # Create a grid of circles
        circles = []
        for i in range(5):
            for j in range(5):
                center = [i * 0.3, j * 0.3]
                circles.append(Hypersphere(center=center, radius=0.2))

        union = UnionDomain(circles)
        assert union.dimension == 2

        # Sample particles
        particles = union.sample_uniform(500, seed=42)
        assert particles.shape[0] <= 500  # May be fewer due to rejection sampling
        assert np.all(union.contains(particles))

    def test_nested_differences(self):
        """Test nested difference operations (Swiss cheese domain)."""
        # Base domain
        base = Hyperrectangle(np.array([[0, 1], [0, 1]]))

        # Multiple holes
        hole1 = Hypersphere(center=[0.25, 0.25], radius=0.1)
        hole2 = Hypersphere(center=[0.75, 0.25], radius=0.1)
        hole3 = Hypersphere(center=[0.25, 0.75], radius=0.1)
        hole4 = Hypersphere(center=[0.75, 0.75], radius=0.1)

        # Create union of holes
        holes = UnionDomain([hole1, hole2, hole3, hole4])

        # Subtract all holes at once
        domain = DifferenceDomain(base, holes)

        # Sample particles
        particles = domain.sample_uniform(500, seed=42)

        # Verify all particles avoid all holes
        for hole in [hole1, hole2, hole3, hole4]:
            assert not np.any(hole.contains(particles))

        # Verify all particles in base domain
        assert np.all(base.contains(particles))


class TestVolumeComputation:
    """Test volume computation for various domains."""

    def test_hyperrectangle_volume_exact(self):
        """Test exact volume computation for hyperrectangles."""
        # 2D rectangle: 3 × 2 = 6
        rect = Hyperrectangle(np.array([[0, 3], [0, 2]]))
        assert np.abs(rect.compute_volume() - 6.0) < 1e-10

        # 3D box: 2 × 3 × 4 = 24
        box = Hyperrectangle(np.array([[0, 2], [0, 3], [0, 4]]))
        assert np.abs(box.compute_volume() - 24.0) < 1e-10

    def test_hypersphere_volume_formula(self):
        """Test hypersphere volume against known formulas."""
        # 2D circle: π r²
        circle = Hypersphere(center=[0, 0], radius=2.0)
        expected_2d = np.pi * 4  # π * 2²
        assert np.abs(circle.compute_volume() - expected_2d) < 0.01

        # 3D sphere: (4/3) π r³
        sphere = Hypersphere(center=[0, 0, 0], radius=2.0)
        expected_3d = (4 / 3) * np.pi * 8  # (4/3) π * 2³
        assert np.abs(sphere.compute_volume() - expected_3d) < 0.01

    def test_difference_domain_volume(self):
        """Test volume of domain with hole."""
        base = Hyperrectangle(np.array([[0, 1], [0, 1]]))  # Volume = 1
        hole = Hypersphere(center=[0.5, 0.5], radius=0.2)
        domain = DifferenceDomain(base, hole)

        # Volume should be: 1 - π * 0.2²
        expected_volume = 1.0 - np.pi * 0.04
        actual_volume = domain.compute_volume(n_monte_carlo=100000)

        # Monte Carlo, so allow 1% error
        assert np.abs(actual_volume - expected_volume) / expected_volume < 0.01


class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_hyperrectangle_bounds(self):
        """Test hyperrectangle with invalid bounds."""
        # min >= max should raise error
        with pytest.raises(ValueError, match="bounds must have min < max"):
            Hyperrectangle(np.array([[1, 0], [0, 1]]))

    def test_invalid_hypersphere_radius(self):
        """Test hypersphere with invalid radius."""
        # Negative radius should raise error
        with pytest.raises(ValueError, match="Radius must be positive"):
            Hypersphere(center=[0, 0], radius=-1.0)

        # Zero radius should raise error
        with pytest.raises(ValueError, match="Radius must be positive"):
            Hypersphere(center=[0, 0], radius=0.0)

    def test_mismatched_dimensions_union(self):
        """Test union with mismatched dimensions."""
        rect_2d = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        rect_3d = Hyperrectangle(np.array([[0, 1], [0, 1], [0, 1]]))

        with pytest.raises(ValueError, match="same dimension"):
            UnionDomain([rect_2d, rect_3d])

    def test_empty_domain_list(self):
        """Test CSG operations with empty domain list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            UnionDomain([])

        with pytest.raises(ValueError, match="cannot be empty"):
            IntersectionDomain([])

    def test_complement_unbounded_sampling(self):
        """Test complement domain without bounding box."""
        circle = Hypersphere(center=[0, 0], radius=1.0)
        exterior = ComplementDomain(circle)

        # Should raise error when trying to get bounding box
        with pytest.raises(ValueError, match="unbounded"):
            exterior.get_bounding_box()

        # Should raise error when trying to sample
        with pytest.raises(ValueError):
            exterior.sample_uniform(100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
