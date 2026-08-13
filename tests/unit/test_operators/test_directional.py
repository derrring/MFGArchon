"""
Unit tests for directional derivative operators and function gradient.

Tests three modules:
- DirectDerivOperator: General directional derivative v dot grad(u)
- NormalDerivOperator: Normal derivative du/dn (specialization)
- function_gradient: Pointwise gradient of callable functions

Created: 2026-02-10 (Issue #768 - Test coverage for operators/)
"""

import pytest

import numpy as np
from scipy.sparse.linalg import LinearOperator

from mfgarchon.operators.differential.directional import (
    DirectDerivOperator,
    NormalDerivOperator,
)
from mfgarchon.operators.differential.function_gradient import (
    HasAnalyticalGradient,
    function_gradient,
    outward_normal_from_sdf,
    sdf_gradient_with_analytical_fallback,
)

# =============================================================================
# Helpers
# =============================================================================


def _1d_grid(n=100, endpoint=True):
    """Create 1D uniform grid on [0, 1]."""
    x = np.linspace(0, 1, n, endpoint=endpoint)
    dx = x[1] - x[0]
    return x, dx


def _2d_grid(nx=50, ny=50, endpoint=True):
    """Create 2D uniform grid on [0, 1]^2."""
    x = np.linspace(0, 1, nx, endpoint=endpoint)
    y = np.linspace(0, 1, ny, endpoint=endpoint)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y, dx, dy


def _sphere_sdf(x):
    """SDF for unit sphere: phi = |x| - 1."""
    x = np.asarray(x)
    if x.ndim == 1:
        return float(np.linalg.norm(x) - 1.0)
    return np.linalg.norm(x, axis=1) - 1.0


def _circle_sdf(x):
    """SDF for unit circle in 2D."""
    x = np.asarray(x)
    if x.ndim == 1:
        return float(np.linalg.norm(x) - 1.0)
    return np.linalg.norm(x, axis=1) - 1.0


def _quadratic_2d(x):
    """f(x, y) = x^2 + 2*y^2, grad = (2x, 4y)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return float(x[0] ** 2 + 2 * x[1] ** 2)
    return x[:, 0] ** 2 + 2 * x[:, 1] ** 2


# =============================================================================
# DirectDerivOperator
# =============================================================================


class TestDirectDerivConstant:
    """Test DirectDerivOperator with constant direction."""

    @pytest.mark.unit
    def test_1d_identity_direction(self):
        """Direction [1.0] should give du/dx."""
        x, dx = _1d_grid(100, endpoint=False)
        u = x**2  # du/dx = 2x

        D = DirectDerivOperator(
            direction=np.array([1.0]),
            spacings=[dx],
            field_shape=(100,),
        )
        du = D(u)

        expected = 2.0 * x
        # Interior: central diff is exact for quadratic
        np.testing.assert_allclose(du[2:-2], expected[2:-2], atol=1e-10)

    @pytest.mark.unit
    def test_1d_scaled_direction(self):
        """Direction [3.0] should give 3 * du/dx."""
        x, dx = _1d_grid(100, endpoint=False)
        u = x**2

        D = DirectDerivOperator(
            direction=np.array([3.0]),
            spacings=[dx],
            field_shape=(100,),
        )
        du = D(u)

        expected = 3.0 * 2.0 * x  # 3 * du/dx
        np.testing.assert_allclose(du[2:-2], expected[2:-2], atol=1e-10)

    @pytest.mark.unit
    def test_2d_constant_direction(self):
        """v = (1, 2), u = x^2 + y^2 => v.grad(u) = 2x + 4y."""
        X, Y, dx, dy = _2d_grid(50, 50, endpoint=False)
        u = X**2 + Y**2

        D = DirectDerivOperator(
            direction=np.array([1.0, 2.0]),
            spacings=[dx, dy],
            field_shape=(50, 50),
        )
        du = D(u)

        expected = 2.0 * X + 4.0 * Y
        # Central difference is exact for polynomial degree <= 2
        np.testing.assert_allclose(du[2:-2, 2:-2], expected[2:-2, 2:-2], atol=1e-10)

    @pytest.mark.unit
    def test_zero_direction_gives_zero(self):
        """Zero direction should produce zero derivative everywhere."""
        x, dx = _1d_grid(50, endpoint=False)
        u = x**3

        D = DirectDerivOperator(
            direction=np.array([0.0]),
            spacings=[dx],
            field_shape=(50,),
        )
        du = D(u)

        np.testing.assert_allclose(du, 0.0, atol=1e-14)


class TestDirectDerivVarying:
    """Test DirectDerivOperator with spatially varying direction."""

    @pytest.mark.unit
    def test_rotation_on_radial(self):
        """Rotation field (y, -x) dotted with grad(x^2 + y^2) = 0.

        For u = x^2 + y^2, grad(u) = (2x, 2y).
        v = (y, -x) is orthogonal to grad(u), so v.grad(u) = 0.
        """
        X, Y, dx, dy = _2d_grid(50, 50, endpoint=False)
        u = X**2 + Y**2
        v_field = np.stack([Y, -X], axis=-1)  # shape (50, 50, 2)

        D = DirectDerivOperator(
            direction=v_field,
            spacings=[dx, dy],
            field_shape=(50, 50),
        )
        du = D(u)

        # Exact for quadratic: central diff exact, dot product is zero
        np.testing.assert_allclose(du[2:-2, 2:-2], 0.0, atol=1e-10)

    @pytest.mark.unit
    def test_varying_matches_constant_when_uniform(self):
        """Uniform varying field should match constant direction."""
        X, Y, dx, dy = _2d_grid(30, 30, endpoint=False)
        u = X**2 + Y

        v_const = np.array([1.0, 0.5])
        # Create "varying" field that is actually constant
        v_field = np.broadcast_to(v_const, (*X.shape, 2)).copy()

        D_const = DirectDerivOperator(direction=v_const, spacings=[dx, dy], field_shape=(30, 30))
        D_vary = DirectDerivOperator(direction=v_field, spacings=[dx, dy], field_shape=(30, 30))

        np.testing.assert_allclose(D_const(u), D_vary(u), atol=1e-14)

    @pytest.mark.unit
    def test_is_constant_property(self):
        """is_constant_direction should distinguish constant vs varying."""
        D_const = DirectDerivOperator(direction=np.array([1.0]), spacings=[0.1], field_shape=(10,))
        assert D_const.is_constant_direction is True

        v_vary = np.ones((10, 1))
        D_vary = DirectDerivOperator(direction=v_vary, spacings=[0.1], field_shape=(10,))
        assert D_vary.is_constant_direction is False


class TestDirectDerivScipyInterface:
    """Test scipy LinearOperator compatibility."""

    @pytest.mark.unit
    def test_isinstance(self):
        """Should be a scipy LinearOperator."""
        D = DirectDerivOperator(
            direction=np.array([1.0, 0.0]),
            spacings=[0.1, 0.1],
            field_shape=(10, 10),
        )
        assert isinstance(D, LinearOperator)

    @pytest.mark.unit
    def test_matvec_callable_consistency(self):
        """D(u) and D @ u.ravel() should agree."""
        X, Y, dx, dy = _2d_grid(20, 20, endpoint=False)
        u = X**2 + Y

        D = DirectDerivOperator(
            direction=np.array([1.0, 2.0]),
            spacings=[dx, dy],
            field_shape=(20, 20),
        )

        du_callable = D(u)
        du_matvec = D @ u.ravel()

        np.testing.assert_allclose(du_callable.ravel(), du_matvec, atol=1e-14)

    @pytest.mark.unit
    def test_operator_shape(self):
        """Shape should be (N, N) where N = prod(field_shape)."""
        D = DirectDerivOperator(
            direction=np.array([1.0, 0.0]),
            spacings=[0.1, 0.1],
            field_shape=(20, 30),
        )
        assert D.shape == (600, 600)

    @pytest.mark.unit
    def test_flattened_input(self):
        """Should accept flattened 1D input."""
        X, _Y, dx, dy = _2d_grid(20, 20, endpoint=False)
        u = X**2

        D = DirectDerivOperator(
            direction=np.array([1.0, 0.0]),
            spacings=[dx, dy],
            field_shape=(20, 20),
        )

        du_flat = D(u.ravel())
        du_field = D(u)

        np.testing.assert_allclose(du_flat, du_field.ravel(), atol=1e-14)


class TestDirectDerivValidation:
    """Test input validation for DirectDerivOperator."""

    @pytest.mark.unit
    def test_direction_dim_mismatch(self):
        """Should raise ValueError for wrong direction dimension."""
        with pytest.raises(ValueError, match="components"):
            DirectDerivOperator(
                direction=np.array([1.0, 2.0]),  # 2D direction
                spacings=[0.1],  # 1D grid
                field_shape=(10,),
            )

    @pytest.mark.unit
    def test_varying_shape_mismatch(self):
        """Should raise ValueError for wrong varying direction shape."""
        with pytest.raises(ValueError, match="doesn't match"):
            DirectDerivOperator(
                direction=np.ones((10, 20, 2)),  # wrong spatial shape
                spacings=[0.1, 0.1],
                field_shape=(10, 10),  # expects (10, 10, 2)
            )

    @pytest.mark.unit
    def test_field_shape_mismatch_callable(self):
        """Should raise ValueError when field shape doesn't match."""
        D = DirectDerivOperator(
            direction=np.array([1.0, 0.0]),
            spacings=[0.1, 0.1],
            field_shape=(10, 10),
        )
        with pytest.raises(ValueError, match="doesn't match"):
            D(np.zeros((10, 20)))

    @pytest.mark.unit
    def test_repr(self):
        """repr should contain key info."""
        D = DirectDerivOperator(
            direction=np.array([1.0, 0.5]),
            spacings=[0.1, 0.1],
            field_shape=(10, 10),
            scheme="upwind",
        )
        r = repr(D)
        assert "DirectDerivOperator" in r
        assert "direction=" in r
        assert "upwind" in r


# =============================================================================
# NormalDerivOperator
# =============================================================================


class TestNormalDerivFromAxis:
    """Test NormalDerivOperator.from_axis() factory method."""

    @pytest.mark.unit
    def test_1d_left_boundary(self):
        """Left boundary normal (-1) on u=x^2 should give du/dn = -2x."""
        x, dx = _1d_grid(100)
        u = x**2

        D_n = NormalDerivOperator.from_axis(axis=0, sign=-1, spacings=[dx], field_shape=(100,))
        du_dn = D_n(u)

        expected = -2.0 * x  # n . grad(u) = -1 * 2x
        np.testing.assert_allclose(du_dn[5:-5], expected[5:-5], atol=1e-10)

    @pytest.mark.unit
    def test_1d_right_boundary(self):
        """Right boundary normal (+1) on u=x^2 should give du/dn = 2x."""
        x, dx = _1d_grid(100)
        u = x**2

        D_n = NormalDerivOperator.from_axis(axis=0, sign=+1, spacings=[dx], field_shape=(100,))
        du_dn = D_n(u)

        expected = 2.0 * x  # n . grad(u) = +1 * 2x
        np.testing.assert_allclose(du_dn[5:-5], expected[5:-5], atol=1e-10)

    @pytest.mark.unit
    def test_2d_y_axis_normal(self):
        """Normal along y-axis (+1) on u = y^2 should give du/dn = 2y."""
        _X, Y, dx, dy = _2d_grid(30, 30)
        u = Y**2

        D_n = NormalDerivOperator.from_axis(axis=1, sign=+1, spacings=[dx, dy], field_shape=(30, 30))
        du_dn = D_n(u)

        expected = 2.0 * Y
        np.testing.assert_allclose(du_dn[2:-2, 2:-2], expected[2:-2, 2:-2], atol=1e-10)

    @pytest.mark.unit
    def test_normal_source_axis(self):
        """normal_source should be 'axis'."""
        D_n = NormalDerivOperator.from_axis(axis=0, sign=1, spacings=[0.1], field_shape=(10,))
        assert D_n.normal_source == "axis"

    @pytest.mark.unit
    def test_axis_out_of_range(self):
        """Should raise ValueError for axis >= ndim."""
        with pytest.raises(ValueError, match="axis"):
            NormalDerivOperator.from_axis(axis=2, sign=1, spacings=[0.1, 0.1], field_shape=(10, 10))

    @pytest.mark.unit
    def test_invalid_sign(self):
        """Should raise ValueError for sign not in {-1, 1}."""
        with pytest.raises(ValueError, match="sign"):
            NormalDerivOperator.from_axis(axis=0, sign=0, spacings=[0.1], field_shape=(10,))


class TestNormalDerivFromSDF:
    """Test NormalDerivOperator.from_sdf() factory method."""

    @pytest.mark.unit
    def test_circle_sdf_normals(self):
        """SDF normals of a circle should point radially outward."""
        nx, ny = 50, 50
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        radius = 0.5
        sdf = np.sqrt(X**2 + Y**2) - radius

        D_n = NormalDerivOperator.from_sdf(sdf, spacings=[dx, dy])

        assert D_n.normal_source == "sdf"
        assert D_n.normal.shape == (nx, ny, 2)

        # Near the circle center, normals should be approximately (x/r, y/r)
        r = np.sqrt(X**2 + Y**2)
        # Check at points not too close to origin
        mask = r > 0.3
        expected_nx = X / np.maximum(r, 1e-10)
        expected_ny = Y / np.maximum(r, 1e-10)

        # Normals should roughly agree with radial direction
        error_x = np.mean(np.abs(D_n.normal[mask, 0] - expected_nx[mask]))
        error_y = np.mean(np.abs(D_n.normal[mask, 1] - expected_ny[mask]))
        assert error_x < 0.1
        assert error_y < 0.1

    @pytest.mark.unit
    def test_from_sdf_applies_correctly(self):
        """SDF-derived operator should compute du/dn on radial function.

        For u = x^2 + y^2 and radial normals n = (x, y)/r:
        du/dn = (x/r)(2x) + (y/r)(2y) = 2r
        """
        nx, ny = 50, 50
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        sdf = np.sqrt(X**2 + Y**2) - 0.5
        D_n = NormalDerivOperator.from_sdf(sdf, spacings=[dx, dy])

        u = X**2 + Y**2
        du_dn = D_n(u)

        r = np.sqrt(X**2 + Y**2)
        expected = 2.0 * r

        # Check away from origin and boundaries
        mask = (r > 0.3) & (np.abs(X) < 0.8) & (np.abs(Y) < 0.8)
        error = np.mean(np.abs(du_dn[mask] - expected[mask]))
        assert error < 0.2


class TestNormalDerivValidation:
    """Test NormalDerivOperator input validation."""

    @pytest.mark.unit
    def test_non_unit_normal_rejected(self):
        """Should raise ValueError for non-unit normal."""
        with pytest.raises(ValueError, match="unit vector"):
            NormalDerivOperator(
                normal=np.array([1.0, 1.0]),  # |n| = sqrt(2) != 1
                spacings=[0.1, 0.1],
                field_shape=(10, 10),
            )

    @pytest.mark.unit
    def test_unit_normal_accepted(self):
        """Should accept properly normalized normals."""
        n = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
        D_n = NormalDerivOperator(normal=n, spacings=[0.1, 0.1], field_shape=(10, 10))
        np.testing.assert_allclose(D_n.normal, n)

    @pytest.mark.unit
    def test_varying_non_unit_rejected(self):
        """Should raise ValueError if any varying normal is not unit length."""
        normals = np.ones((10, 2))  # All have |n| = sqrt(2)
        with pytest.raises(ValueError, match="unit vectors"):
            NormalDerivOperator(normal=normals, spacings=[0.1, 0.1], field_shape=(10,))

    @pytest.mark.unit
    def test_scipy_compatibility(self):
        """NormalDerivOperator should be a scipy LinearOperator."""
        D_n = NormalDerivOperator.from_axis(axis=0, sign=1, spacings=[0.1], field_shape=(20,))
        assert isinstance(D_n, LinearOperator)

        u = np.linspace(0, 1, 20) ** 2
        du_call = D_n(u)
        du_matvec = D_n @ u

        np.testing.assert_allclose(du_call, du_matvec, atol=1e-14)

    @pytest.mark.unit
    def test_repr(self):
        """repr should contain NormalDerivOperator info."""
        D_n = NormalDerivOperator.from_axis(axis=0, sign=-1, spacings=[0.1], field_shape=(10,))
        r = repr(D_n)
        assert "NormalDerivOperator" in r
        assert "axis" in r


# =============================================================================
# function_gradient
# =============================================================================


class TestFunctionGradient:
    """Test pointwise function gradient via finite differences."""

    @pytest.mark.unit
    def test_single_point_sphere(self):
        """Gradient of sphere SDF at (2, 0, 0) should be (1, 0, 0)."""
        point = np.array([2.0, 0.0, 0.0])
        grad = function_gradient(_sphere_sdf, point)

        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(grad, expected, atol=1e-6)

    @pytest.mark.unit
    def test_single_point_shape(self):
        """Single point input should return shape (d,)."""
        grad = function_gradient(_sphere_sdf, np.array([1.0, 0.0, 0.0]))
        assert grad.shape == (3,)

    @pytest.mark.unit
    def test_batch_radial_direction(self):
        """Sphere SDF gradient should point radially outward at all points."""
        points = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )

        grads = function_gradient(_sphere_sdf, points)

        for i, (p, g) in enumerate(zip(points, grads, strict=True)):
            expected_dir = p / np.linalg.norm(p)
            g_normalized = g / np.linalg.norm(g)
            dot = np.dot(g_normalized, expected_dir)
            assert dot > 0.999, f"Point {i}: gradient not radially outward (dot={dot})"

    @pytest.mark.unit
    def test_quadratic_exact(self):
        """Gradient of f(x,y) = x^2 + 2y^2 should be (2x, 4y)."""
        points = np.array([[1.0, 1.0], [2.0, 3.0], [-1.0, 2.0]])
        grads = function_gradient(_quadratic_2d, points)

        expected = np.stack([2 * points[:, 0], 4 * points[:, 1]], axis=1)
        np.testing.assert_allclose(grads, expected, atol=1e-8)

    @pytest.mark.unit
    def test_adaptive_eps_large_coords(self):
        """Adaptive epsilon should handle large coordinates."""
        large_point = np.array([1000.0, 0.0, 0.0])
        grad = function_gradient(_sphere_sdf, large_point, adaptive_eps=True)

        expected_dir = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(grad, expected_dir, atol=1e-4)

    @pytest.mark.unit
    def test_nonfinite_raises(self):
        """Should raise ValueError when function returns non-finite values."""

        def bad_func(x):
            return np.inf

        with pytest.raises(ValueError, match="non-finite"):
            function_gradient(bad_func, np.array([1.0, 0.0]))


class TestOutwardNormal:
    """Test outward_normal_from_sdf."""

    @pytest.mark.unit
    def test_single_point_unit_length(self):
        """Outward normal should be unit length."""
        point = np.array([2.0, 0.0, 0.0])
        normal = outward_normal_from_sdf(_sphere_sdf, point)

        assert abs(np.linalg.norm(normal) - 1.0) < 1e-10

    @pytest.mark.unit
    def test_batch_unit_length(self):
        """All batch normals should be unit length."""
        points = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )

        normals = outward_normal_from_sdf(_sphere_sdf, points)
        norms = np.linalg.norm(normals, axis=1)

        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    @pytest.mark.unit
    def test_circle_normals(self):
        """Normals on unit circle should equal the points themselves.

        For unit circle with SDF |x| - 1, the outward normal at any point
        on the circle (|x| = 1) is just x/|x| = x.
        """
        theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        circle_points = np.stack([np.cos(theta), np.sin(theta)], axis=1)

        normals = outward_normal_from_sdf(_circle_sdf, circle_points)

        np.testing.assert_allclose(normals, circle_points, atol=1e-5)


class TestSDFGradientFallback:
    """Test sdf_gradient_with_analytical_fallback."""

    @pytest.mark.unit
    def test_analytical_preferred(self):
        """Should use analytical gradient when available."""

        class MockGeometry:
            def signed_distance_gradient(self, points):
                return np.ones_like(points) * 42.0

        geom = MockGeometry()
        assert isinstance(geom, HasAnalyticalGradient)

        points = np.array([[1.0, 0.0]])
        grad = sdf_gradient_with_analytical_fallback(geom, points)

        np.testing.assert_allclose(grad, 42.0)

    @pytest.mark.unit
    def test_numerical_fallback_with_sdf_func(self):
        """Should fall back to numerical when no analytical gradient."""

        class PlainGeometry:
            pass

        geom = PlainGeometry()
        points = np.array([[2.0, 0.0]])

        grad = sdf_gradient_with_analytical_fallback(geom, points, sdf_func=_circle_sdf)

        # Gradient at (2, 0) should be approximately (1, 0)
        np.testing.assert_allclose(grad[0, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(grad[0, 1], 0.0, atol=1e-5)

    @pytest.mark.unit
    def test_numerical_fallback_signed_distance_method(self):
        """Should find geometry.signed_distance if no sdf_func provided."""

        class GeomWithSD:
            def signed_distance(self, x):
                return _circle_sdf(x)

        geom = GeomWithSD()
        points = np.array([[2.0, 0.0]])

        grad = sdf_gradient_with_analytical_fallback(geom, points)
        np.testing.assert_allclose(grad[0, 0], 1.0, atol=1e-5)

    @pytest.mark.unit
    def test_no_sdf_raises(self):
        """Should raise ValueError when no SDF function is available."""

        class EmptyGeometry:
            pass

        with pytest.raises(ValueError, match="No SDF"):
            sdf_gradient_with_analytical_fallback(EmptyGeometry(), np.array([[1.0, 0.0]]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
