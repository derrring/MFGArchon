"""
Unit tests for Boundary Condition Value Providers (Issue #625).

Tests the BCValueProvider protocol and concrete implementations for
state-dependent boundary conditions in MFG coupling.
"""

import warnings

import pytest

import numpy as np

from mfgarchon.geometry.boundary.providers import (
    AdjointConsistentProvider,
    BCValueProvider,
    ConstantProvider,
    is_provider,
    resolve_provider,
)

# =============================================================================
# Mock Geometry for Testing
# =============================================================================


class MockGeometry:
    """Mock geometry object for provider tests."""

    def __init__(self, dx: float = 0.1):
        self._dx = dx

    def get_grid_spacing(self) -> list[float]:
        return [self._dx]


# =============================================================================
# Test: BCValueProvider Protocol
# =============================================================================


@pytest.mark.unit
@pytest.mark.fast
class TestBCValueProviderProtocol:
    """Tests for BCValueProvider protocol compliance."""

    def test_adjoint_consistent_provider_is_protocol_compliant(self):
        """AdjointConsistentProvider should implement BCValueProvider protocol."""
        provider = AdjointConsistentProvider(side="left", sigma=0.2)
        assert isinstance(provider, BCValueProvider)

    def test_constant_provider_is_protocol_compliant(self):
        """ConstantProvider should implement BCValueProvider protocol."""
        provider = ConstantProvider(value=1.5)
        assert isinstance(provider, BCValueProvider)

    def test_float_is_not_provider(self):
        """Float values should not be providers."""
        assert not isinstance(1.5, BCValueProvider)
        assert not isinstance(0.0, BCValueProvider)

    def test_is_provider_utility(self):
        """is_provider utility should correctly identify providers."""
        assert is_provider(AdjointConsistentProvider(side="left", sigma=0.2))
        assert is_provider(ConstantProvider(value=1.0))
        assert not is_provider(1.5)
        assert not is_provider(None)
        assert not is_provider(lambda x: x)  # Plain callables are not providers


# =============================================================================
# Test: AdjointConsistentProvider
# =============================================================================


@pytest.mark.unit
@pytest.mark.fast
class TestAdjointConsistentProvider:
    """Tests for AdjointConsistentProvider."""

    def test_init_with_valid_sides(self):
        """Should accept valid side names."""
        # 1D canonical names
        provider_left = AdjointConsistentProvider(side="left", diffusion=0.2)
        assert provider_left.side == "left"

        provider_right = AdjointConsistentProvider(side="right", diffusion=0.2)
        assert provider_right.side == "right"

        # 1D aliases (should normalize to canonical)
        provider_xmin = AdjointConsistentProvider(side="x_min", diffusion=0.2)
        assert provider_xmin.side == "left"  # Normalized

        provider_xmax = AdjointConsistentProvider(side="x_max", diffusion=0.2)
        assert provider_xmax.side == "right"  # Normalized

    def test_init_with_invalid_side_raises(self):
        """Should raise ValueError for invalid side names."""
        with pytest.raises(ValueError, match="side must be one of"):
            AdjointConsistentProvider(side="invalid", diffusion=0.2)

    def test_init_sigma_optional(self):
        """sigma can be None (read from state)."""
        provider = AdjointConsistentProvider(side="left", sigma=None)
        assert provider.sigma is None

    def test_compute_with_exponential_density(self):
        """Test computation with exponential density m(x) = exp(-x).

        For m(x) = exp(-x):
        - d(ln m)/dx = -1 everywhere
        - At left boundary (outward normal is -x): d(ln m)/dn = +1
        - At right boundary (outward normal is +x): d(ln m)/dn = -1
        - g = -diffusion^2/2 * d(ln m)/dn
        """
        diffusion = 0.2
        x = np.linspace(0, 1, 11)
        m = np.exp(-x)

        state = {
            "m_current": m,
            "geometry": MockGeometry(dx=0.1),
            "diffusion": diffusion,
        }

        # Left boundary
        provider_left = AdjointConsistentProvider(side="left", diffusion=diffusion)
        g_left = provider_left.compute(state)
        # d(ln m)/dn_left = +1, so g = -diffusion^2/2 * 1 = -0.02
        expected_left = -(diffusion**2) / 2 * 1.0
        assert abs(g_left - expected_left) < 0.01, f"Left BC: {g_left} != {expected_left}"

        # Right boundary
        provider_right = AdjointConsistentProvider(side="right", diffusion=diffusion)
        g_right = provider_right.compute(state)
        # d(ln m)/dn_right = -1, so g = -diffusion^2/2 * (-1) = 0.02
        expected_right = -(diffusion**2) / 2 * (-1.0)
        assert abs(g_right - expected_right) < 0.01, f"Right BC: {g_right} != {expected_right}"

    def test_compute_reads_diffusion_from_state(self):
        """Should read diffusion from state if not provided in constructor."""
        m = np.exp(-np.linspace(0, 1, 11))
        state = {
            "m_current": m,
            "geometry": MockGeometry(dx=0.1),
            "diffusion": 0.3,  # Diffusion in state
        }

        provider = AdjointConsistentProvider(side="left", diffusion=None)
        g = provider.compute(state)

        # Verify diffusion was read from state
        expected = -(0.3**2) / 2 * 1.0  # Using diffusion=0.3
        assert abs(g - expected) < 0.01

    def test_compute_reads_legacy_sigma_from_state(self):
        """Should read legacy 'sigma' from state for backward compatibility."""
        m = np.exp(-np.linspace(0, 1, 11))
        state = {
            "m_current": m,
            "geometry": MockGeometry(dx=0.1),
            "sigma": 0.3,  # Legacy key
        }

        provider = AdjointConsistentProvider(side="left", diffusion=None)
        g = provider.compute(state)

        # Verify sigma was read from state as fallback
        expected = -(0.3**2) / 2 * 1.0
        assert abs(g - expected) < 0.01

    def test_compute_missing_m_current_raises(self):
        """Should raise KeyError if m_current is missing."""
        state = {
            "geometry": MockGeometry(),
            "diffusion": 0.2,
        }
        provider = AdjointConsistentProvider(side="left", diffusion=0.2)
        with pytest.raises(KeyError, match="m_current"):
            provider.compute(state)

    def test_compute_missing_geometry_raises(self):
        """Should raise KeyError if geometry is missing."""
        state = {
            "m_current": np.ones(10),
            "diffusion": 0.2,
        }
        provider = AdjointConsistentProvider(side="left", diffusion=0.2)
        with pytest.raises(KeyError, match="geometry"):
            provider.compute(state)

    def test_compute_missing_diffusion_raises(self):
        """Should raise KeyError if diffusion is missing and not in constructor."""
        state = {
            "m_current": np.ones(10),
            "geometry": MockGeometry(),
            # No diffusion or sigma
        }
        provider = AdjointConsistentProvider(side="left", diffusion=None)
        with pytest.raises(KeyError, match="diffusion"):
            provider.compute(state)

    def test_repr(self):
        """Test string representation."""
        provider = AdjointConsistentProvider(side="x_min", diffusion=0.2)
        repr_str = repr(provider)
        assert "AdjointConsistentProvider" in repr_str
        assert "x_min" in repr_str  # Should show original side name
        assert "0.2" in repr_str

        provider_no_diff = AdjointConsistentProvider(side="left", diffusion=None)
        repr_str = repr(provider_no_diff)
        assert "from_state" in repr_str

    def test_deprecated_diffusion_parameter(self):
        """Issue #1512: 'diffusion' is now the deprecated alias (a misnomer -- it holds sigma, not the
        PDE D = sigma^2/2); it must still work, emit a deprecation warning, and map to canonical sigma."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            provider = AdjointConsistentProvider(side="left", diffusion=0.2)
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()
            assert provider.sigma == 0.2  # maps to the canonical sigma


# =============================================================================
# Test: ConstantProvider
# =============================================================================


@pytest.mark.unit
@pytest.mark.fast
class TestConstantProvider:
    """Tests for ConstantProvider."""

    def test_compute_returns_constant(self):
        """Should return the constant value regardless of state."""
        provider = ConstantProvider(value=42.0)
        assert provider.compute({}) == 42.0
        assert provider.compute({"anything": "ignored"}) == 42.0

    def test_repr(self):
        """Test string representation."""
        provider = ConstantProvider(value=1.5)
        assert repr(provider) == "ConstantProvider(1.5)"


# =============================================================================
# Test: resolve_provider Utility
# =============================================================================


@pytest.mark.unit
@pytest.mark.fast
class TestResolveProvider:
    """Tests for resolve_provider utility function."""

    def test_resolve_provider_with_provider(self):
        """Should call compute() on providers."""
        provider = ConstantProvider(value=99.0)
        result = resolve_provider(provider, {})
        assert result == 99.0

    def test_resolve_provider_with_float(self):
        """Should return float values unchanged."""
        result = resolve_provider(1.5, {})
        assert result == 1.5

    def test_resolve_provider_with_int(self):
        """Should convert int to float."""
        result = resolve_provider(1, {})
        assert result == 1.0
        assert isinstance(result, float)


# =============================================================================
# Test: Side Name Normalization
# =============================================================================


@pytest.mark.unit
@pytest.mark.fast
class TestSideNameNormalization:
    """Tests for side name normalization in AdjointConsistentProvider."""

    @pytest.mark.parametrize(
        ("input_side", "expected_canonical"),
        [
            ("left", "left"),
            ("right", "right"),
            ("x_min", "left"),
            ("x_max", "right"),
            ("min", "left"),
            ("max", "right"),
            # 2D/3D aliases (for future support)
            ("y_min", "y_min"),
            ("y_max", "y_max"),
            ("bottom", "y_min"),
            ("top", "y_max"),
            ("z_min", "z_min"),
            ("z_max", "z_max"),
            ("front", "z_min"),
            ("back", "z_max"),
        ],
    )
    def test_side_normalization(self, input_side, expected_canonical):
        """Side names should be normalized to canonical form."""
        provider = AdjointConsistentProvider(side=input_side, sigma=0.2)
        assert provider.side == expected_canonical
        assert provider._original_side == input_side


# =============================================================================
# Test: Uncovered Code Paths
# =============================================================================


@pytest.mark.unit
@pytest.mark.fast
class TestProviderEdgeCases:
    """Tests for code paths not covered by main test classes."""

    def test_compute_with_2d_density_takes_last_slice(self):
        """When m has ndim > 1 (time-dependent), compute should use m[-1, :]."""
        # Create 2D density: (T, N) = (5, 11)
        x = np.linspace(0, 1, 11)
        m_2d = np.stack([np.exp(-x) * (1 + 0.1 * t) for t in range(5)])
        assert m_2d.ndim == 2

        state = {
            "m_current": m_2d,
            "geometry": MockGeometry(dx=0.1),
            "diffusion": 0.2,
        }

        # Provider should use m_2d[-1, :] (final time slice)
        provider = AdjointConsistentProvider(side="left", diffusion=0.2)
        g = provider.compute(state)

        # Compute expected from the final slice directly
        m_last = m_2d[-1, :]
        state_1d = {
            "m_current": m_last,
            "geometry": MockGeometry(dx=0.1),
            "diffusion": 0.2,
        }
        g_expected = provider.compute(state_1d)
        assert abs(g - g_expected) < 1e-12

    def test_nd_not_implemented_for_non_lr_side(self):
        """nD geometry with side not in (left, right) should raise NotImplementedError."""

        class MockGeometry2D:
            dimension = 2

            def get_grid_spacing(self):
                return [0.1, 0.1]

        provider = AdjointConsistentProvider(side="y_min", diffusion=0.2)
        state = {
            "m_current": np.ones(10),
            "geometry": MockGeometry2D(),
            "diffusion": 0.2,
        }
        with pytest.raises(NotImplementedError, match="2D not yet implemented"):
            provider.compute(state)

    def test_nd_geometry_with_left_right_uses_1d_path(self):
        """nD geometry with side='left' should fall through to 1D finite diff path."""

        class MockGeometry2D:
            dimension = 2

            def get_grid_spacing(self):
                return [0.1, 0.1]

        x = np.linspace(0, 1, 11)
        m = np.exp(-x)

        provider = AdjointConsistentProvider(side="left", diffusion=0.2)
        state = {
            "m_current": m,
            "geometry": MockGeometry2D(),
            "diffusion": 0.2,
        }
        # Should not raise — falls through to 1D path via OR condition
        g = provider.compute(state)
        expected = -(0.2**2) / 2 * 1.0  # d(ln m)/dn_left ~ 1
        assert abs(g - expected) < 0.01

    def test_sigma_and_diffusion_both_set_sigma_wins(self):
        """Issue #1512: sigma is canonical, so when both are set sigma wins and the deprecated
        diffusion is ignored (passing diffusion at all still triggers the deprecation warning)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            provider = AdjointConsistentProvider(side="left", sigma=0.5, diffusion=0.3)
            # diffusion (deprecated) triggers the warning
            assert len(w) == 1
            # canonical sigma=0.5 wins over the deprecated diffusion=0.3
            assert provider.sigma == 0.5

    def test_resolve_provider_with_ndarray_return(self):
        """resolve_provider should pass through ndarray returns from compute()."""

        class ArrayProvider:
            """Provider that returns an array."""

            def compute(self, state):
                return np.array([1.0, 2.0, 3.0])

        provider = ArrayProvider()
        result = resolve_provider(provider, {})
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
