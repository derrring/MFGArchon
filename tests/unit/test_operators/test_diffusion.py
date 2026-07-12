"""
Unit tests for DiffusionOperator and apply_diffusion.

Tests the unified diffusion operator div(D * grad(u)) that handles (RFC #1596, Path A: the
coefficient is the PDE diffusion tensor D, applied directly with no internal squaring):
- Scalar coefficients: D -> D * Laplacian(u)
- Constant tensor coefficients: D -> div(D * grad(u))
- Spatially varying tensor coefficients: D(x) -> div(D(x) * grad(u))
- from_volatility(sigma): routes sigma -> D through diffusion_from_volatility (Issue #811)

Created: 2026-02-13 (Issue #768 - Test coverage for operators/)
"""

import pytest

import numpy as np
from scipy.sparse.linalg import LinearOperator

from mfgarchon.geometry.boundary import neumann_bc, periodic_bc
from mfgarchon.operators.differential.diffusion import (
    DiffusionOperator,
    apply_diffusion,
)

# =============================================================================
# Fixtures
# =============================================================================


def _1d_grid(n=100):
    """Create 1D uniform grid on [0, 1]."""
    x = np.linspace(0, 1, n)
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
# Scalar (Isotropic) Diffusion
# =============================================================================


class TestIsotropicDiffusion:
    """Tests for scalar coefficient (RFC #1596, Path A): D * Laplacian(u), no internal squaring."""

    @pytest.mark.unit
    def test_1d_quadratic_exact(self):
        """D * Laplacian(x^2) = D * 2 at interior (coefficient is D, not sigma)."""
        x, dx = _1d_grid(100)
        u = x**2
        D_coef = 0.25

        bc = neumann_bc(dimension=1)
        D = DiffusionOperator(coefficient=D_coef, spacings=[dx], field_shape=(100,), bc=bc)
        Du = D(u)

        assert Du.shape == (100,)
        expected = D_coef * 2.0
        error = np.max(np.abs(Du[5:-5] - expected))
        assert error < 1e-8

    @pytest.mark.unit
    def test_from_volatility_scalar_matches_half_sigma_sq(self):
        """from_volatility(sigma) applies D = sigma^2/2 (single-source converter, Issue #811)."""
        x, dx = _1d_grid(100)
        u = x**2
        sigma = 0.5  # D = sigma^2/2 = 0.125

        bc = neumann_bc(dimension=1)
        D = DiffusionOperator.from_volatility(sigma, spacings=[dx], field_shape=(100,), bc=bc)
        Du = D(u)

        expected = (sigma**2 / 2.0) * 2.0  # D * Laplacian(x^2) = 0.125 * 2
        error = np.max(np.abs(Du[5:-5] - expected))
        assert error < 1e-8

    @pytest.mark.unit
    def test_2d_quadratic_exact(self):
        """D * Laplacian(x^2 + y^2) = D * 4 at interior."""
        X, Y, dx, dy = _2d_grid(50, 50)
        u = X**2 + Y**2
        D_coef = 1.0

        bc = neumann_bc(dimension=2)
        D = DiffusionOperator(coefficient=D_coef, spacings=[dx, dy], field_shape=(50, 50), bc=bc)
        Du = D(u)

        assert Du.shape == (50, 50)
        expected = D_coef * 4.0
        error = np.max(np.abs(Du[5:-5, 5:-5] - expected))
        assert error < 1e-8

    @pytest.mark.unit
    def test_1d_constant_zero(self):
        """Diffusion of constant function should be 0."""
        x, dx = _1d_grid(50)
        u = np.ones_like(x) * 3.0

        bc = neumann_bc(dimension=1)
        D = DiffusionOperator(coefficient=2.0, spacings=[dx], field_shape=(50,), bc=bc)
        Du = D(u)

        np.testing.assert_allclose(Du, 0.0, atol=1e-12)

    @pytest.mark.unit
    def test_periodic_sin(self):
        """D * Laplacian(sin(2pi*x)) = -D * 4pi^2 * sin(2pi*x)."""
        n = 100
        x = np.linspace(0, 1, n, endpoint=False)
        dx = x[1] - x[0]
        u = np.sin(2 * np.pi * x)
        D_coef = 1.0

        bc = periodic_bc(dimension=1)
        D = DiffusionOperator(coefficient=D_coef, spacings=[dx], field_shape=(n,), bc=bc)
        Du = D(u)

        expected = -D_coef * ((2 * np.pi) ** 2) * np.sin(2 * np.pi * x)
        error = np.max(np.abs(Du - expected))
        # O(h^2) error for 2nd-order central diff
        assert error < 0.5


# =============================================================================
# Coefficient Type Dispatch
# =============================================================================


class TestCoefficientDispatch:
    """Tests for coefficient processing and type dispatch."""

    @pytest.mark.unit
    def test_scalar_coefficient(self):
        """Scalar coefficient should dispatch to isotropic."""
        D = DiffusionOperator(coefficient=0.5, spacings=[0.1], field_shape=(20,))
        assert D._coeff_type == "scalar"
        assert D.coefficient == 0.5

    @pytest.mark.unit
    def test_scalar_array_coefficient(self):
        """0-d array should be treated as scalar."""
        D = DiffusionOperator(coefficient=np.float64(0.3), spacings=[0.1], field_shape=(20,))
        assert D._coeff_type == "scalar"

    @pytest.mark.unit
    def test_diagonal_vector_coefficient(self):
        """1D vector [sigma_x, sigma_y] should become diagonal tensor."""
        D = DiffusionOperator(
            coefficient=np.array([0.1, 0.2]),
            spacings=[0.1, 0.1],
            field_shape=(20, 20),
        )
        assert D._coeff_type == "constant_tensor"
        expected_diag = np.diag([0.1, 0.2])
        np.testing.assert_allclose(D.coefficient, expected_diag)

    @pytest.mark.unit
    def test_constant_tensor_coefficient(self):
        """(d, d) matrix should be constant tensor."""
        Sigma = np.array([[0.1, 0.02], [0.02, 0.05]])
        D = DiffusionOperator(
            coefficient=Sigma,
            spacings=[0.1, 0.1],
            field_shape=(20, 20),
        )
        assert D._coeff_type == "constant_tensor"

    @pytest.mark.unit
    def test_varying_tensor_coefficient(self):
        """(*field_shape, d, d) array should be varying tensor."""
        Sigma_field = np.zeros((20, 20, 2, 2))
        Sigma_field[..., 0, 0] = 0.1
        Sigma_field[..., 1, 1] = 0.05
        D = DiffusionOperator(
            coefficient=Sigma_field,
            spacings=[0.1, 0.1],
            field_shape=(20, 20),
        )
        assert D._coeff_type == "varying_tensor"

    @pytest.mark.unit
    def test_invalid_coefficient_shape(self):
        """Wrong coefficient shape should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid coefficient shape"):
            DiffusionOperator(
                coefficient=np.ones((3, 3, 3)),
                spacings=[0.1, 0.1],
                field_shape=(20, 20),
            )


# =============================================================================
# Constant Tensor Diffusion
# =============================================================================


class TestConstantTensorDiffusion:
    """Tests for constant tensor coefficient: div(Sigma * grad(u))."""

    @pytest.mark.unit
    def test_diagonal_tensor_matches_weighted_laplacian(self):
        """Diagonal Sigma should give weighted sum of second derivatives.

        For Sigma = diag(a, b) and u = x^2 + y^2:
        div(Sigma * grad(u)) = a * d^2u/dx^2 + b * d^2u/dy^2 = 2a + 2b
        """
        X, Y, dx, dy = _2d_grid(50, 50)
        u = X**2 + Y**2

        a, b = 0.3, 0.1
        Sigma = np.diag([a, b])
        bc = neumann_bc(dimension=2)
        D = DiffusionOperator(coefficient=Sigma, spacings=[dx, dy], field_shape=(50, 50), bc=bc)
        Du = D(u)

        expected = 2 * a + 2 * b  # 0.8
        error = np.max(np.abs(Du[5:-5, 5:-5] - expected))
        assert error < 0.1

    @pytest.mark.unit
    def test_identity_tensor_matches_laplacian(self):
        """Identity tensor should give standard Laplacian."""
        X, _Y, dx, dy = _2d_grid(40, 40)
        u = X**2  # Laplacian = 2

        Sigma = np.eye(2)
        bc = neumann_bc(dimension=2)
        D = DiffusionOperator(coefficient=Sigma, spacings=[dx, dy], field_shape=(40, 40), bc=bc)
        Du = D(u)

        error = np.max(np.abs(Du[5:-5, 5:-5] - 2.0))
        assert error < 0.1

    @pytest.mark.unit
    def test_1d_tensor_scalar_equivalence(self):
        """1D tensor [[D]] should match scalar D diffusion (RFC #1596: coefficient is D on both,
        no branch squares -- this is the scalar==tensor consistency the #1549 flip violated)."""
        x, dx = _1d_grid(80)
        u = x**2
        D_coef = 0.49

        bc = neumann_bc(dimension=1)
        D_scalar = DiffusionOperator(coefficient=D_coef, spacings=[dx], field_shape=(80,), bc=bc)
        D_tensor = DiffusionOperator(
            coefficient=np.array([[D_coef]]),
            spacings=[dx],
            field_shape=(80,),
            bc=bc,
        )

        Du_scalar = D_scalar(u)
        Du_tensor = D_tensor(u)

        # Should match at interior (both compute D * d^2u/dx^2)
        np.testing.assert_allclose(Du_scalar[5:-5], Du_tensor[5:-5], atol=1e-8)


# =============================================================================
# Spatially Varying Tensor Diffusion
# =============================================================================


class TestVaryingTensorDiffusion:
    """Tests for spatially varying tensor: div(Sigma(x) * grad(u))."""

    @pytest.mark.unit
    def test_no_nan_in_output(self):
        """Spatially varying tensor should produce finite results."""
        X, Y, dx, dy = _2d_grid(30, 30)
        u = X**2 + Y**2

        Sigma_field = np.zeros((30, 30, 2, 2))
        Sigma_field[..., 0, 0] = 0.1 * (1 + X)
        Sigma_field[..., 1, 1] = 0.05 * (1 + Y)

        bc = neumann_bc(dimension=2)
        D = DiffusionOperator(coefficient=Sigma_field, spacings=[dx, dy], field_shape=(30, 30), bc=bc)
        Du = D(u)

        assert Du.shape == (30, 30)
        assert np.all(np.isfinite(Du))

    @pytest.mark.unit
    def test_constant_tensor_broadcast_consistency(self):
        """Constant tensor broadcast to field should match varying path.

        When Sigma(x) = Sigma for all x, varying and constant paths should agree.
        """
        X, _Y, dx, dy = _2d_grid(30, 30)
        u = X**2

        Sigma_const = np.array([[0.2, 0.0], [0.0, 0.1]])

        # Constant tensor path
        bc = neumann_bc(dimension=2)
        D_const = DiffusionOperator(
            coefficient=Sigma_const,
            spacings=[dx, dy],
            field_shape=(30, 30),
            bc=bc,
        )

        # Varying tensor (same value everywhere)
        Sigma_vary = np.broadcast_to(Sigma_const, (30, 30, 2, 2)).copy()
        D_vary = DiffusionOperator(
            coefficient=Sigma_vary,
            spacings=[dx, dy],
            field_shape=(30, 30),
            bc=bc,
        )

        Du_const = D_const(u)
        Du_vary = D_vary(u)

        np.testing.assert_allclose(Du_const[3:-3, 3:-3], Du_vary[3:-3, 3:-3], atol=1e-10)

    @pytest.mark.unit
    def test_1d_varying(self):
        """1D varying coefficient: d/dx(sigma(x) * du/dx)."""
        x, dx = _1d_grid(80)
        u = x**2

        # Sigma(x) = [[1+x]]
        Sigma_field = np.zeros((80, 1, 1))
        Sigma_field[:, 0, 0] = 1.0 + x

        bc = neumann_bc(dimension=1)
        D = DiffusionOperator(coefficient=Sigma_field, spacings=[dx], field_shape=(80,), bc=bc)
        Du = D(u)

        assert Du.shape == (80,)
        assert np.all(np.isfinite(Du))


# =============================================================================
# scipy Interface
# =============================================================================


class TestDiffusionScipyInterface:
    """Test scipy LinearOperator compatibility."""

    @pytest.mark.unit
    def test_isinstance(self):
        """Should be a scipy LinearOperator."""
        D = DiffusionOperator(coefficient=0.1, spacings=[0.1], field_shape=(50,))
        assert isinstance(D, LinearOperator)

    @pytest.mark.unit
    def test_operator_shape(self):
        """Operator shape should be (N, N) where N = prod(field_shape)."""
        D = DiffusionOperator(coefficient=0.1, spacings=[0.1, 0.1], field_shape=(20, 30))
        assert D.shape == (600, 600)

    @pytest.mark.unit
    def test_matvec_callable_consistency(self):
        """D(u) and D @ u.ravel() should give identical results."""
        X, _Y, dx, dy = _2d_grid(30, 30)
        u = X**2

        bc = neumann_bc(dimension=2)
        D = DiffusionOperator(coefficient=0.5, spacings=[dx, dy], field_shape=(30, 30), bc=bc)

        Du_callable = D(u)
        Du_matvec = D @ u.ravel()

        np.testing.assert_allclose(Du_callable.ravel(), Du_matvec, atol=1e-14)

    @pytest.mark.unit
    def test_flattened_input(self):
        """Should accept flattened 1D input."""
        X, _Y, dx, dy = _2d_grid(30, 30)
        u = X**2

        bc = neumann_bc(dimension=2)
        D = DiffusionOperator(coefficient=0.5, spacings=[dx, dy], field_shape=(30, 30), bc=bc)

        Du_flat = D(u.ravel())
        Du_field = D(u)

        np.testing.assert_allclose(Du_flat, Du_field.ravel(), atol=1e-14)

    @pytest.mark.unit
    def test_integer_field_shape(self):
        """Should accept integer field_shape for 1D case."""
        D = DiffusionOperator(coefficient=0.1, spacings=[0.1], field_shape=50)
        assert D.field_shape == (50,)
        assert D.shape == (50, 50)


# =============================================================================
# Convenience Function
# =============================================================================


class TestApplyDiffusion:
    """Test apply_diffusion convenience function."""

    @pytest.mark.unit
    def test_matches_operator(self):
        """apply_diffusion should match DiffusionOperator."""
        X, _Y, dx, dy = _2d_grid(30, 30)
        u = X**2

        bc = neumann_bc(dimension=2)
        D = DiffusionOperator(coefficient=0.5, spacings=[dx, dy], field_shape=(30, 30), bc=bc)
        Du_operator = D(u)

        Du_func = apply_diffusion(u, coefficient=0.5, spacings=[dx, dy], bc=bc)

        np.testing.assert_allclose(Du_func, Du_operator, atol=1e-14)


# =============================================================================
# Validation
# =============================================================================


class TestDiffusionValidation:
    """Test input validation."""

    @pytest.mark.unit
    def test_spacings_mismatch(self):
        """Should raise ValueError for mismatched spacings/field_shape."""
        with pytest.raises(ValueError, match="spacings length"):
            DiffusionOperator(coefficient=0.1, spacings=[0.1], field_shape=(10, 10))

    @pytest.mark.unit
    def test_shape_mismatch_callable(self):
        """Should raise ValueError when field shape doesn't match."""
        bc = neumann_bc(dimension=2)
        D = DiffusionOperator(coefficient=0.1, spacings=[0.1, 0.1], field_shape=(10, 10), bc=bc)

        with pytest.raises(ValueError, match="doesn't match"):
            D(np.zeros((10, 20)))

    @pytest.mark.unit
    def test_repr_scalar(self):
        """repr should contain key attributes for scalar coefficient."""
        bc = neumann_bc(dimension=2)
        D = DiffusionOperator(coefficient=0.5, spacings=[0.1, 0.1], field_shape=(10, 10), bc=bc)
        r = repr(D)
        assert "DiffusionOperator" in r
        assert "coefficient=0.5" in r
        assert "field_shape=(10, 10)" in r

    @pytest.mark.unit
    def test_repr_tensor(self):
        """repr should show coefficient_type for tensor coefficient."""
        Sigma = np.eye(2)
        D = DiffusionOperator(coefficient=Sigma, spacings=[0.1, 0.1], field_shape=(10, 10))
        r = repr(D)
        assert "coefficient_type=constant_tensor" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
