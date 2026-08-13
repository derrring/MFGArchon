"""
Tests for SDF utility functions.

Tests cover:
- Basic SDF primitives (sphere, box)
- CSG operations (union, intersection, difference, complement)
- Smooth blending operations
- Gradient computation
- Edge cases and error handling
"""

import pytest

import numpy as np

from mfgarchon.utils.numerical.sdf_utils import (
    sdf_box,
    sdf_complement,
    sdf_difference,
    sdf_gradient,
    sdf_intersection,
    sdf_smooth_intersection,
    sdf_smooth_union,
    sdf_sphere,
    sdf_union,
)


class TestSDFSphere:
    """Test sphere/ball SDF."""

    def test_1d_interval(self):
        """Test 1D case (interval)."""
        points = np.array([0.0, 0.5, 1.0, 1.5])
        dist = sdf_sphere(points, center=[0.5], radius=0.5)

        # Interval centered at 0.5 with radius 0.5 covers [0, 1]
        # point 0.0 is on left boundary, 0.5 is at center (inside), 1.0 is on right boundary, 1.5 is outside
        assert np.abs(dist[0]) < 1e-10  # On left boundary
        assert dist[1] < 0  # At center (inside)
        assert np.abs(dist[2]) < 1e-10  # On right boundary
        assert dist[3] > 0  # Outside

    def test_2d_circle(self):
        """Test 2D circle."""
        points = np.array([[0, 0], [1, 0], [2, 0]])
        dist = sdf_sphere(points, center=[0, 0], radius=1.0)

        assert dist[0] < 0  # Inside (at center)
        assert np.abs(dist[1]) < 1e-10  # On boundary
        assert dist[2] > 0  # Outside

    def test_3d_ball(self):
        """Test 3D ball."""
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        dist = sdf_sphere(points, center=[0, 0, 0], radius=1.0)

        assert dist[0] < 0
        assert np.abs(dist[1]) < 1e-10
        assert dist[2] > 0

    def test_single_point(self):
        """Test single point evaluation."""
        point = np.array([0.5, 0.5])
        dist = sdf_sphere(point, center=[0, 0], radius=1.0)

        assert isinstance(dist, (float, np.floating, np.ndarray))
        assert dist < 0  # Inside unit circle


class TestSDFBox:
    """Test box/hyperrectangle SDF."""

    def test_1d_interval(self):
        """Test 1D interval."""
        points = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        dist = sdf_box(points, bounds=[[0, 1]])

        assert dist[0] > 0  # Outside (left)
        assert np.abs(dist[1]) < 1e-10  # On left boundary
        assert dist[2] < 0  # Inside
        assert np.abs(dist[3]) < 1e-10  # On right boundary
        assert dist[4] > 0  # Outside (right)

    def test_2d_rectangle(self):
        """Test 2D rectangle."""
        points = np.array([[0.5, 0.5], [0, 0], [2, 2], [-1, -1]])
        dist = sdf_box(points, bounds=[[0, 1], [0, 1]])

        assert dist[0] < 0  # Inside (center)
        assert np.abs(dist[1]) < 1e-10  # On corner
        assert dist[2] > 0  # Outside
        assert dist[3] > 0  # Outside

    def test_3d_box(self):
        """Test 3D box."""
        points = np.array([[0.5, 0.5, 0.5], [0, 0, 0], [2, 2, 2]])
        dist = sdf_box(points, bounds=[[0, 1], [0, 1], [0, 1]])

        assert dist[0] < 0  # Inside
        assert np.abs(dist[1]) < 1e-10  # On corner
        assert dist[2] > 0  # Outside


class TestSDFUnion:
    """Test union operation."""

    def test_two_circles(self):
        """Test union of two circles."""
        points = np.linspace(-2, 2, 100).reshape(-1, 1)

        sdf1 = sdf_sphere(points, center=[-0.5], radius=0.5)
        sdf2 = sdf_sphere(points, center=[0.5], radius=0.5)
        union = sdf_union(sdf1, sdf2)

        # Union should be inside if either circle is inside
        assert np.all(union <= sdf1)  # Union less restrictive
        assert np.all(union <= sdf2)

        # At center of left circle, should match left circle
        idx_left = np.argmin(np.abs(points[:, 0] + 0.5))
        assert np.abs(union[idx_left] - sdf1[idx_left]) < 1e-10

    def test_multiple_sdfs(self):
        """Test union of 3+ SDFs."""
        points = np.array([[0, 0], [1, 1], [-1, -1]])

        sdf1 = sdf_sphere(points, center=[0, 0], radius=0.5)
        sdf2 = sdf_sphere(points, center=[1, 1], radius=0.5)
        sdf3 = sdf_sphere(points, center=[-1, -1], radius=0.5)

        union = sdf_union(sdf1, sdf2, sdf3)

        # Each point should be inside at least one sphere
        assert union[0] < 0  # Inside first sphere
        assert union[1] < 0  # Inside second sphere
        assert union[2] < 0  # Inside third sphere

    def test_empty_raises(self):
        """Test that empty union raises error."""
        with pytest.raises(ValueError, match="At least one SDF required"):
            sdf_union()


class TestSDFIntersection:
    """Test intersection operation."""

    def test_two_circles(self):
        """Test intersection of two circles."""
        points = np.linspace(-2, 2, 100).reshape(-1, 1)

        sdf1 = sdf_sphere(points, center=[-0.3], radius=0.7)
        sdf2 = sdf_sphere(points, center=[0.3], radius=0.7)
        intersection = sdf_intersection(sdf1, sdf2)

        # Intersection should be more restrictive
        assert np.all(intersection >= sdf1)
        assert np.all(intersection >= sdf2)

        # Depends on exact geometry, but intersection should be valid
        # (no specific test needed here, just that it's more restrictive)

    def test_box_and_circle(self):
        """Test box intersected with circle."""
        points = np.random.uniform(-2, 2, (100, 2))

        box_dist = sdf_box(points, bounds=[[-1, 1], [-1, 1]])
        sphere_dist = sdf_sphere(points, center=[0, 0], radius=0.8)
        intersection = sdf_intersection(box_dist, sphere_dist)

        # Points inside intersection must be inside both
        inside_intersection = intersection < 0
        if np.any(inside_intersection):
            assert np.all(box_dist[inside_intersection] < 0)
            assert np.all(sphere_dist[inside_intersection] < 0)


class TestSDFComplement:
    """Test complement operation."""

    def test_sphere_complement(self):
        """Test complement of sphere (exterior)."""
        points = np.array([[0, 0], [1, 0], [2, 0]])
        sphere_dist = sdf_sphere(points, center=[0, 0], radius=1.0)

        complement = sdf_complement(sphere_dist)

        # Signs should be flipped
        assert complement[0] > 0  # Was inside, now outside
        assert np.abs(complement[1]) < 1e-10  # Boundary unchanged
        assert complement[2] < 0  # Was outside, now inside

    def test_double_complement(self):
        """Test that complement(complement(sdf)) == sdf."""
        points = np.random.uniform(-2, 2, (100, 2))
        sdf = sdf_sphere(points, center=[0, 0], radius=1.0)

        double_complement = sdf_complement(sdf_complement(sdf))

        np.testing.assert_allclose(double_complement, sdf, atol=1e-10)


class TestSDFDifference:
    """Test difference operation."""

    def test_box_with_hole(self):
        """Test box with circular hole."""
        points = np.random.uniform(-2, 2, (200, 2))

        box = sdf_box(points, bounds=[[-1, 1], [-1, 1]])
        hole = sdf_sphere(points, center=[0, 0], radius=0.5)
        domain = sdf_difference(box, hole)

        # Points inside domain must be inside box AND outside hole
        inside_domain = domain < 0
        if np.any(inside_domain):
            assert np.all(box[inside_domain] < 0)  # Inside box
            assert np.all(hole[inside_domain] > 0)  # Outside hole


class TestSmoothOperations:
    """Test smooth blending operations."""

    def test_smooth_union(self):
        """Test smooth union creates smooth blend."""
        points = np.linspace(-2, 2, 100).reshape(-1, 1)

        sdf1 = sdf_sphere(points, center=[-0.5], radius=0.5)
        sdf2 = sdf_sphere(points, center=[0.5], radius=0.5)

        # Sharp union
        sharp = sdf_union(sdf1, sdf2)

        # Smooth union
        smooth = sdf_smooth_union(sdf1, sdf2, smoothing=0.2)

        # Smooth should be close to sharp but slightly different
        assert not np.allclose(smooth, sharp)  # Different

        # Smooth union should blend between the two inputs
        # Check that it's bounded by the inputs (approximately)
        assert np.all(smooth <= np.maximum(sdf1, sdf2) + 0.3)  # Smoothing adds some distance
        assert np.all(smooth >= sharp - 0.3)  # But stays close to minimum

    def test_smooth_intersection(self):
        """Test smooth intersection."""
        points = np.linspace(-2, 2, 100).reshape(-1, 1)

        sdf1 = sdf_sphere(points, center=[-0.3], radius=0.7)
        sdf2 = sdf_sphere(points, center=[0.3], radius=0.7)

        sharp = sdf_intersection(sdf1, sdf2)
        smooth = sdf_smooth_intersection(sdf1, sdf2, smoothing=0.1)

        # Smooth should be close to sharp but slightly different
        assert not np.allclose(smooth, sharp)
        # Smooth intersection is less restrictive (smaller values = more inside)
        assert np.all(smooth <= sharp + 1e-10)


class TestSDFGradient:
    """Test gradient computation."""

    def test_sphere_gradient_points_outward(self):
        """Test that sphere gradient points radially outward."""
        # Points on x-axis
        points = np.array([[0.5, 0], [-0.5, 0], [0, 0.5], [0, -0.5]])

        def sdf_func(p):
            return sdf_sphere(p, center=[0, 0], radius=1.0)

        grad = sdf_gradient(points, sdf_func, epsilon=1e-5)

        # Gradient should point radially outward (normalized position vector)
        expected = points / np.linalg.norm(points, axis=1, keepdims=True)

        np.testing.assert_allclose(grad, expected, atol=1e-3)

    def test_box_gradient(self):
        """Test box gradient."""
        points = np.array([[0.5, 0.5]])  # Center of unit box

        def sdf_func(p):
            return sdf_box(p, bounds=[[0, 1], [0, 1]])

        grad = sdf_gradient(points, sdf_func, epsilon=1e-5)

        # At center, gradient should be roughly zero (or pointing to nearest edge)
        assert grad.shape == (1, 2)  # One point, 2D gradient

    def test_single_point_gradient(self):
        """Test gradient for single point."""
        point = np.array([0.5, 0])

        def sdf_func(p):
            return sdf_sphere(p, center=[0, 0], radius=1.0)

        grad = sdf_gradient(point, sdf_func)

        assert grad.shape == (2,)
        # Should point outward (normalized radial direction)
        # For point at (0.5, 0), normalized direction is (1, 0)
        assert np.abs(grad[0] - 1.0) < 1e-2
        assert np.abs(grad[1]) < 1e-2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_points(self):
        """Test with empty point array."""
        points = np.array([]).reshape(0, 2)
        dist = sdf_sphere(points, center=[0, 0], radius=1.0)
        assert dist.shape == (0,)

    def test_high_dimensional(self):
        """Test with high-dimensional points."""
        # 5D hypersphere
        points = np.random.randn(10, 5)
        center = np.zeros(5)
        dist = sdf_sphere(points, center=center, radius=1.0)

        assert dist.shape == (10,)

    def test_list_inputs(self):
        """Test that lists are converted correctly."""
        points_list = [[0, 0], [1, 0]]
        center_list = [0, 0]

        dist = sdf_sphere(points_list, center=center_list, radius=1.0)

        assert isinstance(dist, np.ndarray)
        assert dist.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
