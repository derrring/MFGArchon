"""
Unit tests for NumericalScheme enum.

Tests the numerical scheme enumeration used in the three-mode solving API
for Issue #580 (adjoint-aware solver pairing).
"""

from mfgarchon.types import NumericalScheme


class TestNumericalSchemeEnum:
    """Test NumericalScheme enum definition and values."""

    def test_enum_values_exist(self):
        """Test that all expected enum values are defined."""
        assert hasattr(NumericalScheme, "FDM_UPWIND")
        assert hasattr(NumericalScheme, "FDM_CENTERED")
        assert hasattr(NumericalScheme, "SL_LINEAR")
        assert hasattr(NumericalScheme, "SL_CUBIC")
        assert hasattr(NumericalScheme, "GFDM")

    def test_enum_string_values(self):
        """Test that enum values match expected strings."""
        assert NumericalScheme.FDM_UPWIND.value == "fdm_upwind"
        assert NumericalScheme.FDM_CENTERED.value == "fdm_centered"
        assert NumericalScheme.SL_LINEAR.value == "sl_linear"
        assert NumericalScheme.SL_CUBIC.value == "sl_cubic"
        assert NumericalScheme.GFDM.value == "gfdm"

    def test_enum_str_representation(self):
        """Test __str__ method returns value string."""
        assert str(NumericalScheme.FDM_UPWIND) == "fdm_upwind"
        assert str(NumericalScheme.GFDM) == "gfdm"

    def test_enum_equality(self):
        """Test enum equality comparison."""
        assert NumericalScheme.FDM_UPWIND == NumericalScheme.FDM_UPWIND
        assert NumericalScheme.FDM_UPWIND != NumericalScheme.GFDM

    def test_enum_iteration(self):
        """Test that enum can be iterated."""
        schemes = list(NumericalScheme)
        assert len(schemes) == 8
        assert NumericalScheme.FDM_UPWIND in schemes
        assert NumericalScheme.FEM_P1 in schemes
        assert NumericalScheme.GFDM in schemes
        assert NumericalScheme.MESHLESS_GALERKIN in schemes


class TestDiscreteDuality:
    """Test is_discrete_dual() classification method."""

    def test_fdm_has_discrete_duality(self):
        """Test that FDM schemes have discrete duality (Type A)."""
        assert NumericalScheme.FDM_UPWIND.is_discrete_dual()
        assert NumericalScheme.FDM_CENTERED.is_discrete_dual()

    def test_sl_has_discrete_duality(self):
        """Test that Semi-Lagrangian schemes have discrete duality (Type A)."""
        assert NumericalScheme.SL_LINEAR.is_discrete_dual()
        assert NumericalScheme.SL_CUBIC.is_discrete_dual()

    def test_gfdm_has_continuous_duality(self):
        """Test that GFDM has only continuous duality (Type B)."""
        assert not NumericalScheme.GFDM.is_discrete_dual()

    def test_meshless_galerkin_has_discrete_duality(self):
        """Test that meshless Galerkin (MLS) has discrete duality (Type A; Issue #1131)."""
        assert NumericalScheme.MESHLESS_GALERKIN.is_discrete_dual()

    def test_discrete_dual_count(self):
        """Test that exactly 7 schemes have discrete duality (FDM, SL, FEM, MESHLESS_GALERKIN)."""
        discrete_schemes = [s for s in NumericalScheme if s.is_discrete_dual()]
        assert len(discrete_schemes) == 7

    def test_continuous_dual_count(self):
        """Test that exactly 1 scheme has continuous duality."""
        continuous_schemes = [s for s in NumericalScheme if not s.is_discrete_dual()]
        assert len(continuous_schemes) == 1
        assert continuous_schemes[0] == NumericalScheme.GFDM


class TestMassRenormalization:
    """Test requires_renormalization() classification method."""

    def test_fdm_no_renormalization(self):
        """Test that FDM schemes don't require renormalization."""
        assert not NumericalScheme.FDM_UPWIND.requires_renormalization()
        assert not NumericalScheme.FDM_CENTERED.requires_renormalization()

    def test_sl_no_renormalization(self):
        """Test that Semi-Lagrangian schemes don't require renormalization."""
        assert not NumericalScheme.SL_LINEAR.requires_renormalization()
        assert not NumericalScheme.SL_CUBIC.requires_renormalization()

    def test_gfdm_requires_renormalization(self):
        """Test that GFDM requires renormalization."""
        assert NumericalScheme.GFDM.requires_renormalization()

    def test_renormalization_inverse_of_duality(self):
        """Test that renormalization is needed exactly when discrete duality is absent."""
        for scheme in NumericalScheme:
            assert scheme.requires_renormalization() == (not scheme.is_discrete_dual())


class TestExperimentalStatus:
    """Test is_experimental property classification."""

    def test_fdm_not_experimental(self):
        """Test that FDM schemes are production-ready."""
        assert not NumericalScheme.FDM_UPWIND.is_experimental
        assert not NumericalScheme.FDM_CENTERED.is_experimental

    def test_sl_linear_not_experimental(self):
        """Test that SL_LINEAR is production-ready."""
        assert not NumericalScheme.SL_LINEAR.is_experimental

    def test_sl_cubic_is_experimental(self):
        """Test that SL_CUBIC is experimental (Issue #583)."""
        assert NumericalScheme.SL_CUBIC.is_experimental

    def test_gfdm_not_experimental(self):
        """Test that GFDM is production-ready (despite Type B duality)."""
        assert not NumericalScheme.GFDM.is_experimental

    def test_experimental_count(self):
        """Test that exactly 1 scheme is experimental."""
        experimental = [s for s in NumericalScheme if s.is_experimental]
        assert len(experimental) == 1
        assert experimental[0] == NumericalScheme.SL_CUBIC


class TestImportability:
    """Test that enum is properly exported and importable."""

    def test_import_from_types(self):
        """Test import from mfgarchon.types."""
        from mfgarchon.types import NumericalScheme

        assert NumericalScheme.FDM_UPWIND.value == "fdm_upwind"

    def test_import_from_main(self):
        """Test import from main mfgarchon module."""
        from mfgarchon import NumericalScheme

        assert NumericalScheme.GFDM.value == "gfdm"

    def test_no_circular_imports(self):
        """Test that importing doesn't cause circular import errors."""
        # This test passes if no ImportError is raised
        from mfgarchon.types.schemes import NumericalScheme

        assert NumericalScheme is not None


class TestEnumDocumentation:
    """Test that enum has comprehensive docstrings."""

    def test_class_has_docstring(self):
        """Test that NumericalScheme class has docstring."""
        assert NumericalScheme.__doc__ is not None
        assert len(NumericalScheme.__doc__) > 100  # Comprehensive docstring

    def test_methods_have_docstrings(self):
        """Test that helper methods have docstrings."""
        assert NumericalScheme.is_discrete_dual.__doc__ is not None
        assert NumericalScheme.requires_renormalization.__doc__ is not None
        # is_experimental is a property, check its docstring
        assert NumericalScheme.is_experimental.fget.__doc__ is not None


class TestEnumUsagePatterns:
    """Test common usage patterns."""

    def test_enum_in_dictionary(self):
        """Test that enum can be used as dictionary key."""
        scheme_info = {
            NumericalScheme.FDM_UPWIND: "first-order",
            NumericalScheme.SL_LINEAR: "second-order",
        }
        assert scheme_info[NumericalScheme.FDM_UPWIND] == "first-order"

    def test_enum_in_set(self):
        """Test that enum can be used in set."""
        fdm_schemes = {NumericalScheme.FDM_UPWIND, NumericalScheme.FDM_CENTERED}
        assert NumericalScheme.FDM_UPWIND in fdm_schemes
        assert NumericalScheme.GFDM not in fdm_schemes

    def test_enum_sorting(self):
        """Test that enums can be sorted (by value)."""
        schemes = [
            NumericalScheme.GFDM,
            NumericalScheme.FDM_UPWIND,
            NumericalScheme.SL_LINEAR,
        ]
        # Sorting by value string
        sorted_schemes = sorted(schemes, key=lambda s: s.value)
        assert sorted_schemes[0] == NumericalScheme.FDM_UPWIND  # "fdm_upwind"
        assert sorted_schemes[-1] == NumericalScheme.SL_LINEAR  # "sl_linear"

    def test_enum_list_comprehension(self):
        """Test enum usage in list comprehension."""
        production_schemes = [s for s in NumericalScheme if not s.is_experimental]
        assert len(production_schemes) == 7
        assert NumericalScheme.SL_CUBIC not in production_schemes


if __name__ == "__main__":
    # Smoke test - run basic checks
    print("Running NumericalScheme enum smoke tests...")

    # Test enum values
    print(f"✓ FDM_UPWIND: {NumericalScheme.FDM_UPWIND.value}")
    print(f"✓ GFDM: {NumericalScheme.GFDM.value}")

    # Test classification methods
    print(f"✓ FDM has discrete duality: {NumericalScheme.FDM_UPWIND.is_discrete_dual()}")
    print(f"✓ GFDM requires renormalization: {NumericalScheme.GFDM.requires_renormalization()}")
    print(f"✓ SL_CUBIC is experimental: {NumericalScheme.SL_CUBIC.is_experimental}")

    # Test iteration
    print(f"✓ Total schemes: {len(list(NumericalScheme))}")

    # Test importability
    from mfgarchon import NumericalScheme as NumScheme

    print(f"✓ Import from main package: {NumScheme.FDM_UPWIND.value}")

    print("\nAll smoke tests passed! ✓")
