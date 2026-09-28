"""
Unit tests for adjoint BC coupling (mfgarchon/alg/numerical/adjoint/bc_coupling.py).

Tests the core functions used by AdjointConsistentProvider to compute
boundary log-density gradients and create coupled Robin BC objects.

References:
    - Issue #574: Adjoint-consistent BC implementation
    - Issue #704: Unified adjoint module
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.adjoint.bc_coupling import (
    compute_adjoint_consistent_bc_values,
    compute_boundary_log_density_gradient_1d,
    compute_coupled_hjb_bc_values,
    create_adjoint_consistent_bc_1d,
)
from mfgarchon.geometry.boundary.types import BCType

# =============================================================================
# Mock Geometry
# =============================================================================


class _MockGeometry1D:
    """Minimal geometry for testing (1D)."""

    dimension = 1
    domain_bounds = np.array([[0.0, 1.0]])

    def __init__(self, dx: float = 0.1, n_points: int = 11):
        self._dx = dx
        self._n = n_points

    def get_grid_spacing(self) -> list[float]:
        return [self._dx]


# =============================================================================
# Test: compute_boundary_log_density_gradient_1d
# =============================================================================


@pytest.mark.unit
@pytest.mark.fast
class TestBoundaryLogDensityGradient1D:
    """Tests for 1D boundary log-density gradient computation."""

    def test_exponential_density_left(self):
        """For m(x) = exp(-x), d(ln m)/dn at left boundary should be +1.

        ln(exp(-x)) = -x, so d/dx = -1.
        Outward normal at left is -x, so d/dn = -(-1) = +1.
        """
        x = np.linspace(0, 1, 101)  # Fine grid for accuracy
        m = np.exp(-x)
        dx = x[1] - x[0]

        grad = compute_boundary_log_density_gradient_1d(m, dx, "left")
        assert abs(grad - 1.0) < 1e-4

    def test_exponential_density_right(self):
        """For m(x) = exp(-x), d(ln m)/dn at right boundary should be -1.

        d/dx[ln(exp(-x))] = -1, outward normal at right is +x.
        d/dn = -1.
        """
        x = np.linspace(0, 1, 101)
        m = np.exp(-x)
        dx = x[1] - x[0]

        grad = compute_boundary_log_density_gradient_1d(m, dx, "right")
        assert abs(grad - (-1.0)) < 1e-4

    def test_uniform_density_zero_gradient(self):
        """Uniform density should give zero gradient at both boundaries."""
        m = np.ones(51)
        dx = 0.02

        grad_left = compute_boundary_log_density_gradient_1d(m, dx, "left")
        grad_right = compute_boundary_log_density_gradient_1d(m, dx, "right")

        assert abs(grad_left) < 1e-8
        assert abs(grad_right) < 1e-8

    def test_invalid_side_raises(self):
        """Invalid side name should raise ValueError."""
        m = np.ones(10)
        with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
            compute_boundary_log_density_gradient_1d(m, 0.1, "top")

    def test_regularization_prevents_log_zero(self):
        """Density with zeros should not raise due to regularization."""
        m = np.zeros(11)  # All zeros
        dx = 0.1
        # Should not raise — regularization prevents log(0)
        grad = compute_boundary_log_density_gradient_1d(m, dx, "left")
        # With uniform m=0 + reg, gradient should be ~0
        assert np.isfinite(grad)
        assert abs(grad) < 1e-6

    def test_gaussian_density(self):
        """Gaussian density m(x) = exp(-x^2) has known gradients.

        ln(m) = -x^2, d/dx = -2x.
        Left (x=0): d/dx = 0, d/dn = 0.
        Right (x=1): d/dx = -2, d/dn = -2.
        """
        x = np.linspace(0, 1, 201)
        m = np.exp(-(x**2))
        dx = x[1] - x[0]

        grad_left = compute_boundary_log_density_gradient_1d(m, dx, "left")
        grad_right = compute_boundary_log_density_gradient_1d(m, dx, "right")

        # At x=0: d(ln m)/dx = 0, outward = -x, so d/dn = 0
        assert abs(grad_left - 0.0) < 0.02

        # At x=1: d(ln m)/dx = -2, outward = +x, so d/dn = -2
        assert abs(grad_right - (-2.0)) < 0.02

    def test_return_type_is_float(self):
        """Return value should always be a Python float."""
        m = np.ones(11)
        grad = compute_boundary_log_density_gradient_1d(m, 0.1, "left")
        assert isinstance(grad, float)


# =============================================================================
# Test: create_adjoint_consistent_bc_1d
# =============================================================================


@pytest.mark.unit
@pytest.mark.fast
class TestCreateAdjointConsistentBC1D:
    """Tests for 1D adjoint-consistent Robin BC creation."""

    def test_creates_two_robin_segments(self):
        """Should create BoundaryConditions with 2 Robin segments."""
        m = np.exp(-np.linspace(0, 1, 11))
        bc = create_adjoint_consistent_bc_1d(m, dx=0.1, volatility=0.2)

        assert bc.dimension == 1
        assert len(bc.segments) == 2
        for seg in bc.segments:
            assert seg.bc_type == BCType.ROBIN

    def test_robin_coefficients(self):
        """Robin segments should have alpha=0, beta=1 (pure Neumann form)."""
        m = np.ones(11)
        bc = create_adjoint_consistent_bc_1d(m, dx=0.1, volatility=0.2)

        for seg in bc.segments:
            assert seg.alpha == 0.0
            assert seg.beta == 1.0

    def test_segment_names_and_boundaries(self):
        """Segments should be named and assigned to correct boundaries."""
        m = np.ones(11)
        bc = create_adjoint_consistent_bc_1d(m, dx=0.1, volatility=0.2)

        left = bc.segments[0]
        right = bc.segments[1]

        assert left.name == "left_adjoint_consistent"
        assert left.boundary == "x_min"
        assert right.name == "right_adjoint_consistent"
        assert right.boundary == "x_max"

    def test_values_with_exponential_density(self):
        """BC values should match -sigma^2/2 * d(ln m)/dn for exponential density."""
        x = np.linspace(0, 1, 101)
        m = np.exp(-x)
        dx = x[1] - x[0]
        sigma = 0.2

        bc = create_adjoint_consistent_bc_1d(m, dx=dx, volatility=sigma)

        # Expected: g = -sigma^2/2 * d(ln m)/dn
        # Left: d(ln m)/dn ~ 1.0, so g ~ -0.02
        # Right: d(ln m)/dn ~ -1.0, so g ~ 0.02
        diffusion_coeff = sigma**2 / 2
        assert abs(bc.segments[0].value - (-diffusion_coeff * 1.0)) < 0.001
        assert abs(bc.segments[1].value - (-diffusion_coeff * (-1.0))) < 0.001

    def test_domain_bounds_passed_through(self):
        """Domain bounds should be passed to BoundaryConditions."""
        m = np.ones(11)
        bounds = np.array([[0.0, 2.0]])
        bc = create_adjoint_consistent_bc_1d(m, dx=0.2, volatility=0.1, domain_bounds=bounds)

        assert bc.domain_bounds is not None
        np.testing.assert_array_equal(bc.domain_bounds, bounds)

    def test_uniform_density_gives_zero_values(self):
        """Uniform density should produce zero BC values (zero gradient)."""
        m = np.ones(51)
        bc = create_adjoint_consistent_bc_1d(m, dx=0.02, volatility=0.3)

        for seg in bc.segments:
            assert abs(seg.value) < 1e-8


# =============================================================================
# Test: compute_adjoint_consistent_bc_values (dimension dispatch)
# =============================================================================


@pytest.mark.unit
@pytest.mark.fast
class TestComputeAdjointConsistentBCValues:
    """Tests for dimension-dispatching BC creation."""

    def test_1d_dispatch(self):
        """1D should produce valid Robin BC identical to direct 1D call."""
        m = np.exp(-np.linspace(0, 1, 11))
        geom = _MockGeometry1D(dx=0.1)

        bc = compute_adjoint_consistent_bc_values(m, geom, volatility=0.2, dimension=1)
        bc_direct = create_adjoint_consistent_bc_1d(m, dx=0.1, volatility=0.2)

        assert len(bc.segments) == len(bc_direct.segments)
        for seg_a, seg_b in zip(bc.segments, bc_direct.segments, strict=False):
            assert abs(seg_a.value - seg_b.value) < 1e-12

    def test_nd_raises_not_implemented(self):
        """Dimensions > 1 should raise NotImplementedError."""
        m = np.ones(25)
        geom = _MockGeometry1D()

        with pytest.raises(NotImplementedError, match="not yet implemented for 2D"):
            compute_adjoint_consistent_bc_values(m, geom, volatility=0.2, dimension=2)

        with pytest.raises(NotImplementedError, match="not yet implemented for 3D"):
            compute_adjoint_consistent_bc_values(m, geom, volatility=0.2, dimension=3)

    def test_backward_compat_alias(self):
        """compute_coupled_hjb_bc_values should be an alias."""
        assert compute_coupled_hjb_bc_values is compute_adjoint_consistent_bc_values
