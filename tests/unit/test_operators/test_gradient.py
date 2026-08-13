"""
Unit tests for PartialDerivOperator and GradientOperator.

Tests first-order derivative operators on structured grids using known
analytical solutions where central differences are exact (linear, quadratic)
and known-accuracy cases (trig functions).

Created: 2026-02-10 (Issue #768 - Test coverage for operators/)
"""

import pytest

import numpy as np
from scipy.sparse.linalg import LinearOperator

from mfgarchon.geometry.boundary import neumann_bc, no_flux_bc
from mfgarchon.operators.differential.gradient import (
    GradientOperator,
    PartialDerivOperator,
)

# =============================================================================
# Fixtures
# =============================================================================


def _1d_grid(n=100):
    """Create 1D uniform grid on [0, 2pi]."""
    x = np.linspace(0, 2 * np.pi, n)
    dx = x[1] - x[0]
    return x, dx


def _2d_grid(nx=50, ny=50):
    """Create 2D uniform grid on [0, 1]^2."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y, dx, dy


# =============================================================================
# PartialDerivOperator Tests
# =============================================================================


class TestPartialDerivOperator:
    """Tests for PartialDerivOperator (single partial derivative)."""

    @pytest.mark.unit
    def test_1d_linear_exact(self):
        """Central difference of linear function u=3x+1 should be exact."""
        x, dx = _1d_grid(100)
        u = 3.0 * x + 1.0

        d_dx = PartialDerivOperator(direction=0, spacings=[dx], field_shape=(100,))
        du_dx = d_dx(u)

        # Central difference is exact for linear functions (interior)
        # Boundary wraps around due to no BC, so check interior only
        assert np.max(np.abs(du_dx[2:-2] - 3.0)) < 1e-10

    @pytest.mark.unit
    def test_1d_sin_accuracy(self):
        """Central difference of sin(x) should approximate cos(x) with O(h^2)."""
        x, dx = _1d_grid(200)
        u = np.sin(x)

        d_dx = PartialDerivOperator(direction=0, spacings=[dx], field_shape=(200,))
        du_dx = d_dx(u)

        expected = np.cos(x)
        # Interior error should be O(h^2) ~ 5e-4 for h ~ 0.03
        error = np.max(np.abs(du_dx[5:-5] - expected[5:-5]))
        assert error < 1e-2

    @pytest.mark.unit
    def test_scipy_linear_operator_interface(self):
        """Should implement scipy LinearOperator and give consistent results."""
        x, dx = _1d_grid(50)
        u = np.sin(x)

        d_dx = PartialDerivOperator(direction=0, spacings=[dx], field_shape=(50,))

        assert isinstance(d_dx, LinearOperator)
        assert d_dx.shape == (50, 50)

        # Callable and matvec should give identical results
        du_callable = d_dx(u)
        du_matvec = d_dx @ u.ravel()
        np.testing.assert_allclose(du_callable.ravel(), du_matvec, atol=1e-14)

    @pytest.mark.unit
    def test_2d_quadratic_exact_x(self):
        """Partial d/dx of u=x^2+y should give exactly 2x (quadratic is exact for central diff)."""
        X, Y, dx, dy = _2d_grid(50, 50)
        u = X**2 + Y

        d_dx = PartialDerivOperator(direction=0, spacings=[dx, dy], field_shape=(50, 50))
        du_dx = d_dx(u)

        expected = 2.0 * X
        # Quadratic: central difference is exact in interior
        error = np.max(np.abs(du_dx[2:-2, 2:-2] - expected[2:-2, 2:-2]))
        assert error < 1e-10

    @pytest.mark.unit
    def test_2d_quadratic_exact_y(self):
        """Partial d/dy of u=x+y^2 should give exactly 2y."""
        X, Y, dx, dy = _2d_grid(50, 50)
        u = X + Y**2

        d_dy = PartialDerivOperator(direction=1, spacings=[dx, dy], field_shape=(50, 50))
        du_dy = d_dy(u)

        expected = 2.0 * Y
        error = np.max(np.abs(du_dy[2:-2, 2:-2] - expected[2:-2, 2:-2]))
        assert error < 1e-10

    @pytest.mark.unit
    def test_2d_shape_preserved(self):
        """2D output shape should match field_shape."""
        X, Y, dx, dy = _2d_grid(40, 30)
        u = X + Y

        d_dx = PartialDerivOperator(direction=0, spacings=[dx, dy], field_shape=(40, 30))
        assert d_dx(u).shape == (40, 30)

    @pytest.mark.unit
    def test_with_neumann_bc(self):
        """Should work with Neumann BC (no-flux)."""
        X, _Y, dx, dy = _2d_grid(50, 50)
        u = X**2

        bc = neumann_bc(dimension=2)
        d_dx = PartialDerivOperator(direction=0, spacings=[dx, dy], field_shape=(50, 50), bc=bc)
        du_dx = d_dx(u)

        assert du_dx.shape == (50, 50)
        # Interior should still be accurate
        expected = 2.0 * X
        error = np.max(np.abs(du_dx[5:-5, 5:-5] - expected[5:-5, 5:-5]))
        assert error < 1e-10

    @pytest.mark.unit
    def test_upwind_scheme(self):
        """Upwind scheme should converge (1st order) for monotone function."""
        x, dx = _1d_grid(200)
        u = x**2  # Monotone increasing on [0, 2pi]

        d_dx = PartialDerivOperator(direction=0, spacings=[dx], field_shape=(200,), scheme="upwind")
        du_dx = d_dx(u)

        expected = 2.0 * x
        # Upwind is 1st order - larger error than central
        error = np.max(np.abs(du_dx[5:-5] - expected[5:-5]))
        assert error < 0.5  # O(h) ~ 0.03

    @pytest.mark.unit
    def test_one_sided_scheme(self):
        """One-sided scheme should use forward/backward at boundaries."""
        x, dx = _1d_grid(100)
        u = x**2

        d_dx = PartialDerivOperator(direction=0, spacings=[dx], field_shape=(100,), scheme="one_sided")
        du_dx = d_dx(u)

        # Interior should be central-difference accurate
        expected = 2.0 * x
        error = np.max(np.abs(du_dx[5:-5] - expected[5:-5]))
        assert error < 1e-10  # Quadratic still exact for central

    @pytest.mark.unit
    def test_direction_validation(self):
        """Should raise ValueError for invalid direction."""
        with pytest.raises(ValueError, match="direction 2 >= dimension 2"):
            PartialDerivOperator(direction=2, spacings=[0.1, 0.1], field_shape=(10, 10))

    @pytest.mark.unit
    def test_spacings_validation(self):
        """Should raise ValueError for mismatched spacings."""
        with pytest.raises(ValueError, match="spacings length"):
            PartialDerivOperator(direction=0, spacings=[0.1], field_shape=(10, 10))

    @pytest.mark.unit
    def test_repr(self):
        """repr should include key information."""
        d_dx = PartialDerivOperator(direction=0, spacings=[0.1, 0.1], field_shape=(10, 10))
        r = repr(d_dx)
        assert "PartialDerivOperator" in r
        assert "d/dx" in r
        assert "field_shape=(10, 10)" in r


# =============================================================================
# GradientOperator Tests
# =============================================================================


class TestGradientOperator:
    """Tests for GradientOperator (full gradient)."""

    @pytest.mark.unit
    def test_1d_output_shape(self):
        """1D gradient should return shape (N, 1)."""
        x, dx = _1d_grid(80)
        u = np.sin(x)

        grad = GradientOperator(spacings=[dx], field_shape=(80,))
        grad_u = grad(u)

        assert grad_u.shape == (80, 1)

    @pytest.mark.unit
    def test_2d_output_shape(self):
        """2D gradient should return shape (Nx, Ny, 2)."""
        X, Y, dx, dy = _2d_grid(40, 30)
        u = X + Y

        grad = GradientOperator(spacings=[dx, dy], field_shape=(40, 30))
        grad_u = grad(u)

        assert grad_u.shape == (40, 30, 2)

    @pytest.mark.unit
    def test_2d_linear_gradient(self):
        """Gradient of u=3x+2y should be (3, 2) everywhere (interior)."""
        X, Y, dx, dy = _2d_grid(50, 50)
        u = 3.0 * X + 2.0 * Y

        grad = GradientOperator(spacings=[dx, dy], field_shape=(50, 50))
        grad_u = grad(u)

        # Linear function: central difference is exact
        error_x = np.max(np.abs(grad_u[2:-2, 2:-2, 0] - 3.0))
        error_y = np.max(np.abs(grad_u[2:-2, 2:-2, 1] - 2.0))
        assert error_x < 1e-10
        assert error_y < 1e-10

    @pytest.mark.unit
    def test_2d_quadratic_gradient(self):
        """Gradient of u=x^2+y^3 should be (2x, 3y^2)."""
        X, Y, dx, dy = _2d_grid(50, 50)
        u = X**2 + Y**3

        grad = GradientOperator(spacings=[dx, dy], field_shape=(50, 50))
        grad_u = grad(u)

        # x^2 component: exact for central diff
        error_x = np.max(np.abs(grad_u[2:-2, 2:-2, 0] - 2.0 * X[2:-2, 2:-2]))
        assert error_x < 1e-10

        # y^3 component: O(h^2) error
        expected_y = 3.0 * Y**2
        error_y = np.max(np.abs(grad_u[5:-5, 5:-5, 1] - expected_y[5:-5, 5:-5]))
        assert error_y < 1e-3

    @pytest.mark.unit
    def test_magnitude(self):
        """Gradient magnitude |nabla u| for u=x should be 1.0."""
        X, _Y, dx, dy = _2d_grid(50, 50)
        u = X  # grad = (1, 0), |grad| = 1

        grad = GradientOperator(spacings=[dx, dy], field_shape=(50, 50))
        mag = grad.magnitude(u)

        assert mag.shape == (50, 50)
        # Interior: |grad(x)| = 1
        error = np.max(np.abs(mag[2:-2, 2:-2] - 1.0))
        assert error < 1e-10

    @pytest.mark.unit
    def test_component_access(self):
        """Accessing components should give same result as full gradient slices."""
        X, Y, dx, dy = _2d_grid(40, 40)
        u = X**2 + Y

        grad = GradientOperator(spacings=[dx, dy], field_shape=(40, 40))
        grad_u = grad(u)

        # Component access should match full gradient
        du_dx = grad.components[0](u)
        du_dy = grad.components[1](u)

        np.testing.assert_allclose(grad_u[..., 0], du_dx, atol=1e-14)
        np.testing.assert_allclose(grad_u[..., 1], du_dy, atol=1e-14)

    @pytest.mark.unit
    def test_3d_linear(self):
        """3D gradient of u=x+2y+3z should be (1, 2, 3)."""
        Nx, Ny, Nz = 20, 20, 20
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        z = np.linspace(0, 1, Nz)
        dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        u = X + 2.0 * Y + 3.0 * Z

        grad = GradientOperator(spacings=[dx, dy, dz], field_shape=(Nx, Ny, Nz))
        grad_u = grad(u)

        assert grad_u.shape == (Nx, Ny, Nz, 3)

        s = slice(3, -3)
        assert np.max(np.abs(grad_u[s, s, s, 0] - 1.0)) < 1e-10
        assert np.max(np.abs(grad_u[s, s, s, 1] - 2.0)) < 1e-10
        assert np.max(np.abs(grad_u[s, s, s, 2] - 3.0)) < 1e-10

    @pytest.mark.unit
    def test_flattened_input(self):
        """Should accept flattened (1D) input and reshape internally."""
        X, Y, dx, dy = _2d_grid(30, 30)
        u = X + Y

        grad = GradientOperator(spacings=[dx, dy], field_shape=(30, 30))
        grad_u = grad(u.ravel())

        assert grad_u.shape == (30, 30, 2)

    @pytest.mark.unit
    def test_with_bc(self):
        """Should work with boundary conditions."""
        X, _Y, dx, dy = _2d_grid(40, 40)
        u = X**2

        bc = no_flux_bc(dimension=2)
        grad = GradientOperator(spacings=[dx, dy], field_shape=(40, 40), bc=bc)
        grad_u = grad(u)

        assert grad_u.shape == (40, 40, 2)
        # Interior d/dx should be 2x
        expected = 2.0 * X
        error = np.max(np.abs(grad_u[5:-5, 5:-5, 0] - expected[5:-5, 5:-5]))
        assert error < 1e-10

    @pytest.mark.unit
    def test_spacings_validation(self):
        """Should raise ValueError for mismatched spacings."""
        with pytest.raises(ValueError, match="spacings length"):
            GradientOperator(spacings=[0.1], field_shape=(10, 10))

    @pytest.mark.unit
    def test_repr(self):
        """repr should include key information."""
        grad = GradientOperator(spacings=[0.1, 0.1], field_shape=(10, 10))
        r = repr(grad)
        assert "GradientOperator" in r
        assert "ndim=2" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
