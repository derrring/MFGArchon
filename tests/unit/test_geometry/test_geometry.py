"""
Quick tests for implicit domain geometry infrastructure.

Run with: python -m pytest tests/unit/geometry/test_geometry.py -v
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


class TestHyperrectangle:
    """Test hyperrectangle domain."""

    def test_2d_unit_square(self):
        """Test 2D unit square [0,1]²."""
        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))

        assert domain.dimension == 2
        assert domain.contains(np.array([0.5, 0.5]))  # Interior
        assert domain.contains(np.array([1.0, 1.0]))  # Boundary
        assert not domain.contains(np.array([1.5, 0.5]))  # Exterior

    def test_4d_hypercube(self):
        """Test 4D unit hypercube."""
        domain = Hyperrectangle(np.array([[0, 1]] * 4))

        assert domain.dimension == 4
        particles = domain.sample_uniform(1000, seed=42)
        assert particles.shape == (1000, 4)
        assert np.all(domain.contains(particles))

    def test_signed_distance(self):
        """Test exact signed distance function."""
        domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))

        # Interior: should be negative
        assert domain.signed_distance(np.array([0.5, 0.5])) < 0

        # Boundary: should be zero
        assert np.abs(domain.signed_distance(np.array([1.0, 0.5]))) < 1e-10

        # Exterior: should be positive
        assert domain.signed_distance(np.array([1.5, 0.5])) > 0

    def test_volume(self):
        """Test volume calculation."""
        domain = Hyperrectangle(np.array([[0, 2], [0, 3]]))
        assert np.abs(domain.compute_volume() - 6.0) < 1e-10  # 2 × 3


class TestHypersphere:
    """Test hypersphere domain."""

    def test_2d_unit_circle(self):
        """Test 2D unit circle."""
        domain = Hypersphere(center=[0, 0], radius=1.0)

        assert domain.dimension == 2
        assert domain.contains(np.array([0, 0]))  # Center
        assert domain.contains(np.array([0.5, 0]))  # Interior
        assert not domain.contains(np.array([2, 0]))  # Exterior

    def test_radius_must_be_positive_and_finite(self):
        """Issue #1077: a non-positive or non-finite radius must fail fast. inf/nan slip
        past the `<= 0` check and would otherwise yield inf bounding boxes -> NaN in
        rejection sampling."""
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="positive"):
                Hypersphere(center=[0, 0], radius=bad)
        for bad in (float("inf"), float("nan")):
            with pytest.raises(ValueError, match="finite"):
                Hypersphere(center=[0, 0], radius=bad)

    def test_4d_hypersphere(self):
        """Test 4D hypersphere."""
        domain = Hypersphere(center=[0] * 4, radius=1.0)

        assert domain.dimension == 4
        particles = domain.sample_uniform(1000, seed=42)
        assert particles.shape == (1000, 4)
        assert np.all(domain.contains(particles))

    def test_signed_distance(self):
        """Test exact signed distance function."""
        domain = Hypersphere(center=[0, 0], radius=1.0)

        # Center: should be -radius
        assert np.abs(domain.signed_distance(np.array([0, 0])) + 1.0) < 1e-10

        # Boundary: should be zero
        assert np.abs(domain.signed_distance(np.array([1, 0]))) < 1e-10

        # Exterior: should be positive
        assert domain.signed_distance(np.array([2, 0])) > 0

    def test_volume_2d(self):
        """Test volume (area) of 2D circle."""
        circle = Hypersphere(center=[0, 0], radius=1.0)
        volume = circle.compute_volume()
        assert np.abs(volume - np.pi) < 1e-6

    def test_volume_3d(self):
        """Test volume of 3D sphere."""
        sphere = Hypersphere(center=[0, 0, 0], radius=1.0)
        volume = sphere.compute_volume()
        expected = 4 * np.pi / 3
        assert np.abs(volume - expected) < 1e-6


class TestBoundaryNormals:
    """Test boundary normal computation and projection."""

    def test_sphere_boundary_normal_radial(self):
        """Test sphere normal points radially outward."""
        sphere = Hypersphere(center=[0, 0], radius=1.0)

        # Point on boundary at (1, 0) - normal should be (1, 0)
        normal = sphere.get_boundary_normal(np.array([1.0, 0.0]))
        assert np.allclose(normal, [1.0, 0.0], atol=1e-6)

        # Point on boundary at (0, 1) - normal should be (0, 1)
        normal = sphere.get_boundary_normal(np.array([0.0, 1.0]))
        assert np.allclose(normal, [0.0, 1.0], atol=1e-6)

        # Point on boundary at (1/sqrt(2), 1/sqrt(2)) - normal should be same
        sqrt2_inv = 1.0 / np.sqrt(2)
        normal = sphere.get_boundary_normal(np.array([sqrt2_inv, sqrt2_inv]))
        assert np.allclose(normal, [sqrt2_inv, sqrt2_inv], atol=1e-6)

    def test_sphere_boundary_normal_interior(self):
        """Test sphere normal from interior point."""
        sphere = Hypersphere(center=[0, 0], radius=1.0)

        # Interior point at (0.5, 0) - normal still points radially
        normal = sphere.get_boundary_normal(np.array([0.5, 0.0]))
        assert np.allclose(normal, [1.0, 0.0], atol=1e-6)

    def test_sphere_boundary_normal_batch(self):
        """Test sphere normal computation for batch of points."""
        sphere = Hypersphere(center=[0, 0], radius=1.0)

        points = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        normals = sphere.get_boundary_normal(points)

        assert normals.shape == (4, 2)
        assert np.allclose(normals[0], [1.0, 0.0], atol=1e-6)
        assert np.allclose(normals[1], [0.0, 1.0], atol=1e-6)
        assert np.allclose(normals[2], [-1.0, 0.0], atol=1e-6)
        assert np.allclose(normals[3], [0.0, -1.0], atol=1e-6)

    def test_rectangle_boundary_normal_faces(self):
        """Test rectangle normal on each face."""
        rect = Hyperrectangle(np.array([[0, 1], [0, 1]]))

        # Right face at x=1 - normal should be (1, 0)
        normal = rect.get_boundary_normal(np.array([1.0, 0.5]))
        assert np.allclose(normal, [1.0, 0.0], atol=1e-6)

        # Top face at y=1 - normal should be (0, 1)
        normal = rect.get_boundary_normal(np.array([0.5, 1.0]))
        assert np.allclose(normal, [0.0, 1.0], atol=1e-6)

        # Left face at x=0 - normal should be (-1, 0)
        normal = rect.get_boundary_normal(np.array([0.0, 0.5]))
        assert np.allclose(normal, [-1.0, 0.0], atol=1e-6)

        # Bottom face at y=0 - normal should be (0, -1)
        normal = rect.get_boundary_normal(np.array([0.5, 0.0]))
        assert np.allclose(normal, [0.0, -1.0], atol=1e-6)

    def test_project_to_boundary_sphere(self):
        """Test projection to sphere boundary."""
        sphere = Hypersphere(center=[0, 0], radius=1.0)

        # Interior point
        inside = np.array([0.5, 0.0])
        projected = sphere.project_to_boundary(inside)
        assert np.allclose(projected, [1.0, 0.0], atol=1e-6)
        assert sphere.is_on_boundary(projected)

        # Exterior point
        outside = np.array([2.0, 0.0])
        projected = sphere.project_to_boundary(outside)
        assert np.allclose(projected, [1.0, 0.0], atol=1e-6)
        assert sphere.is_on_boundary(projected)

        # Diagonal point
        diag_inside = np.array([0.3, 0.3])
        projected = sphere.project_to_boundary(diag_inside)
        # Should be on boundary (norm = 1)
        assert np.abs(np.linalg.norm(projected) - 1.0) < 1e-6
        assert sphere.is_on_boundary(projected)

    def test_project_to_boundary_batch(self):
        """Test projection for batch of points."""
        sphere = Hypersphere(center=[0, 0], radius=1.0)

        points = np.array(
            [
                [0.3, 0.0],  # Interior
                [1.5, 0.0],  # Exterior
                [0.0, 0.4],  # Interior
                [0.0, 2.0],  # Exterior
            ]
        )
        projected = sphere.project_to_boundary(points)

        assert projected.shape == (4, 2)
        # All should be on boundary (norm = 1)
        norms = np.linalg.norm(projected, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
        # All should be detected as on boundary
        assert np.all(sphere.is_on_boundary(projected))

    def test_is_on_boundary_sphere(self):
        """Test boundary detection for sphere."""
        sphere = Hypersphere(center=[0, 0], radius=1.0)

        # On boundary
        assert sphere.is_on_boundary(np.array([1.0, 0.0]))
        assert sphere.is_on_boundary(np.array([0.0, 1.0]))

        # Not on boundary (interior)
        assert not sphere.is_on_boundary(np.array([0.5, 0.0]))
        assert not sphere.is_on_boundary(np.array([0.0, 0.0]))

        # Not on boundary (exterior)
        assert not sphere.is_on_boundary(np.array([1.5, 0.0]))

    def test_is_on_boundary_batch(self):
        """Test boundary detection for batch of points."""
        sphere = Hypersphere(center=[0, 0], radius=1.0)

        points = np.array(
            [
                [1.0, 0.0],  # On boundary
                [0.5, 0.0],  # Interior
                [0.0, 1.0],  # On boundary
                [1.5, 0.0],  # Exterior
            ]
        )
        on_boundary = sphere.is_on_boundary(points)

        assert on_boundary.shape == (4,)
        assert on_boundary[0]  # On boundary
        assert not on_boundary[1]  # Interior
        assert on_boundary[2]  # On boundary
        assert not on_boundary[3]  # Exterior

    def test_high_dimensional_normals(self):
        """Test boundary normals in higher dimensions."""
        # 4D hypersphere
        sphere = Hypersphere(center=[0, 0, 0, 0], radius=1.0)

        # Point on boundary along first axis
        point = np.array([1.0, 0.0, 0.0, 0.0])
        normal = sphere.get_boundary_normal(point)

        assert normal.shape == (4,)
        assert np.allclose(normal, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_projection_convergence(self):
        """Test that projection converges to boundary."""
        sphere = Hypersphere(center=[0, 0], radius=1.0)

        # Random interior and exterior points
        np.random.seed(42)
        interior = np.random.uniform(-0.9, 0.9, (10, 2))
        exterior = np.random.uniform(1.1, 2.0, (10, 2))

        # All projections should end up on boundary
        proj_interior = sphere.project_to_boundary(interior)
        proj_exterior = sphere.project_to_boundary(exterior)

        # Check all are on boundary
        for proj in proj_interior:
            assert np.abs(np.linalg.norm(proj) - 1.0) < 1e-6

        for proj in proj_exterior:
            assert np.abs(np.linalg.norm(proj) - 1.0) < 1e-6


class TestCSGOperations:
    """Test CSG operations."""

    def test_union_2_circles(self):
        """Test union of two circles."""
        circle1 = Hypersphere(center=[0, 0], radius=1.0)
        circle2 = Hypersphere(center=[1, 0], radius=1.0)
        union = UnionDomain([circle1, circle2])

        # Point in overlap region
        assert union.contains(np.array([0.5, 0]))

        # Points in individual circles
        assert union.contains(np.array([-0.5, 0]))  # Only in circle1
        assert union.contains(np.array([1.5, 0]))  # Only in circle2

        # Point outside both
        assert not union.contains(np.array([3, 0]))

    def test_intersection_rect_circle(self):
        """Test intersection of rectangle and circle."""
        rect = Hyperrectangle(np.array([[0, 2], [0, 2]]))
        circle = Hypersphere(center=[1, 1], radius=0.8)  # Smaller radius
        intersection = IntersectionDomain([rect, circle])

        # Point in both
        assert intersection.contains(np.array([1, 1]))

        # Point in rect but not circle
        assert not intersection.contains(np.array([0.1, 0.1]))

        # Point in circle but not rect
        assert not intersection.contains(np.array([-0.5, 1]))

    def test_complement(self):
        """Test complement of circle."""
        circle = Hypersphere(center=[0, 0], radius=1.0)
        exterior = ComplementDomain(circle)

        # Inside circle → outside complement
        assert not exterior.contains(np.array([0, 0]))

        # Outside circle → inside complement
        assert exterior.contains(np.array([2, 0]))

    def test_difference_rect_minus_circle(self):
        """Test rectangle with circular hole."""
        rect = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        hole = Hypersphere(center=[0.5, 0.5], radius=0.2)
        domain = DifferenceDomain(rect, hole)

        # Point in rectangle but outside hole
        assert domain.contains(np.array([0.1, 0.1]))

        # Point in hole
        assert not domain.contains(np.array([0.5, 0.5]))

        # Point outside rectangle
        assert not domain.contains(np.array([1.5, 0.5]))

    def test_complex_domain_obstacle_avoidance(self):
        """Test domain with obstacle (key for Stage 2!)."""
        # Base: unit square
        base = Hyperrectangle(np.array([[0, 1], [0, 1]]))

        # Obstacle: circular
        obstacle = Hypersphere(center=[0.5, 0.5], radius=0.2)

        # Navigable domain
        domain = DifferenceDomain(base, obstacle)

        # Sample particles
        particles = domain.sample_uniform(1000, seed=42)

        # Verify all particles avoid obstacle
        dists = np.linalg.norm(particles - np.array([0.5, 0.5]), axis=1)
        assert np.all(dists > 0.2)  # All particles outside obstacle

        # Verify all particles in base domain
        assert np.all((particles >= 0) & (particles <= 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
