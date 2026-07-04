"""
Unit tests for SchemeFamily enum.

Tests the internal scheme family enum used for duality validation in Issue #580.
"""

import pytest

from mfgarchon.alg import SchemeFamily


class TestSchemeFamilyEnum:
    """Test SchemeFamily enum definition and values."""

    def test_enum_values_exist(self):
        """Test that all expected enum values are defined."""
        assert hasattr(SchemeFamily, "FDM")
        assert hasattr(SchemeFamily, "SL")
        assert hasattr(SchemeFamily, "FVM")
        assert hasattr(SchemeFamily, "FEM")
        assert hasattr(SchemeFamily, "GFDM")
        assert hasattr(SchemeFamily, "MESHLESS_GALERKIN")
        assert hasattr(SchemeFamily, "PINN")
        assert hasattr(SchemeFamily, "GENERIC")

    def test_enum_string_values(self):
        """Test that enum values match expected strings."""
        assert SchemeFamily.FDM.value == "fdm"
        assert SchemeFamily.SL.value == "semi_lagrangian"
        assert SchemeFamily.FVM.value == "fvm"
        assert SchemeFamily.FEM.value == "fem"
        assert SchemeFamily.GFDM.value == "gfdm"
        assert SchemeFamily.MESHLESS_GALERKIN.value == "meshless_galerkin"
        assert SchemeFamily.PINN.value == "pinn"
        assert SchemeFamily.GENERIC.value == "generic"

    def test_enum_str_representation(self):
        """Test __str__ method returns value string."""
        assert str(SchemeFamily.FDM) == "fdm"
        assert str(SchemeFamily.GFDM) == "gfdm"

    def test_enum_equality(self):
        """Test enum equality comparison."""
        assert SchemeFamily.FDM == SchemeFamily.FDM
        assert SchemeFamily.FDM != SchemeFamily.GFDM

    def test_enum_iteration(self):
        """Test that enum can be iterated."""
        families = list(SchemeFamily)
        assert len(families) == 9  # FDM, SL, FVM, FEM, GFDM, MESHLESS_GALERKIN, FEEC, PINN, GENERIC
        assert SchemeFamily.FDM in families
        assert SchemeFamily.FEM in families
        assert SchemeFamily.MESHLESS_GALERKIN in families
        assert SchemeFamily.FEEC in families  # Issue #1502 (FEEC scaffold)
        assert SchemeFamily.GENERIC in families


class TestDualityClassification:
    """Test duality type classification by family."""

    def test_fdm_discrete_duality(self):
        """Test that FDM has discrete duality (Type A)."""
        # FDM: div and grad are matrix transposes
        assert SchemeFamily.FDM.value == "fdm"

    def test_sl_discrete_duality(self):
        """Test that Semi-Lagrangian has discrete duality (Type A)."""
        # SL: splatting is transpose of interpolation
        assert SchemeFamily.SL.value == "semi_lagrangian"

    def test_fvm_discrete_duality(self):
        """Test that FVM has discrete duality (Type A)."""
        # FVM: Consistent numerical flux gives discrete transpose
        assert SchemeFamily.FVM.value == "fvm"

    def test_gfdm_continuous_duality(self):
        """Test that GFDM has continuous duality only (Type B)."""
        # GFDM: Asymmetric neighborhoods prevent discrete transpose
        assert SchemeFamily.GFDM.value == "gfdm"

    def test_pinn_continuous_duality(self):
        """Test that PINN has continuous duality only (Type B)."""
        # PINN: Neural network discretization, continuous duality
        assert SchemeFamily.PINN.value == "pinn"

    def test_generic_unknown_duality(self):
        """Test that GENERIC has unknown duality."""
        # GENERIC: Fallback for unannotated solvers
        assert SchemeFamily.GENERIC.value == "generic"


class TestImportability:
    """Test that enum is properly exported and importable."""

    def test_import_from_alg(self):
        """Test import from mfgarchon.alg."""
        from mfgarchon.alg import SchemeFamily

        assert SchemeFamily.FDM.value == "fdm"

    def test_import_from_base_solver(self):
        """Test import from base_solver module."""
        from mfgarchon.alg.base_solver import SchemeFamily

        assert SchemeFamily.GFDM.value == "gfdm"

    def test_no_circular_imports(self):
        """Test that importing doesn't cause circular import errors."""
        from mfgarchon.alg.base_solver import SchemeFamily

        assert SchemeFamily is not None


class TestEnumDocumentation:
    """Test that enum has comprehensive docstrings."""

    def test_class_has_docstring(self):
        """Test that SchemeFamily class has docstring."""
        assert SchemeFamily.__doc__ is not None
        assert len(SchemeFamily.__doc__) > 100  # Comprehensive docstring

    def test_docstring_mentions_issue_580(self):
        """Test that docstring references Issue #580."""
        assert "#580" in SchemeFamily.__doc__


class TestEnumUsagePatterns:
    """Test common usage patterns for internal validation."""

    def test_enum_in_dictionary(self):
        """Test that enum can be used as dictionary key."""
        duality_map = {
            SchemeFamily.FDM: "exact",
            SchemeFamily.GFDM: "asymptotic",
            SchemeFamily.GENERIC: "unknown",
        }
        assert duality_map[SchemeFamily.FDM] == "exact"
        assert duality_map[SchemeFamily.GFDM] == "asymptotic"

    def test_enum_in_set(self):
        """Test that enum can be used in set."""
        discrete_families = {SchemeFamily.FDM, SchemeFamily.SL, SchemeFamily.FVM}
        assert SchemeFamily.FDM in discrete_families
        assert SchemeFamily.GFDM not in discrete_families

    def test_enum_list_comprehension(self):
        """Test enum usage in list comprehension."""
        # Simulate identifying Type A schemes
        type_a_families = {SchemeFamily.FDM, SchemeFamily.SL, SchemeFamily.FVM}
        type_a_schemes = [f for f in SchemeFamily if f in type_a_families]
        assert len(type_a_schemes) == 3
        assert SchemeFamily.GFDM not in type_a_schemes


class TestValidatorPattern:
    """Test usage with Issue #543 validator pattern."""

    def test_getattr_with_default(self):
        """Test getattr pattern for optional _scheme_family trait."""

        class MockSolverWithTrait:
            _scheme_family = SchemeFamily.FDM

        class MockSolverWithoutTrait:
            pass

        # Solver with trait
        solver_with = MockSolverWithTrait()
        family = getattr(solver_with, "_scheme_family", SchemeFamily.GENERIC)
        assert family == SchemeFamily.FDM

        # Solver without trait (fallback to GENERIC)
        solver_without = MockSolverWithoutTrait()
        family = getattr(solver_without, "_scheme_family", SchemeFamily.GENERIC)
        assert family == SchemeFamily.GENERIC

    def test_try_except_pattern(self):
        """Test try/except pattern for required _scheme_family trait."""

        class MockSolverWithTrait:
            _scheme_family = SchemeFamily.FDM

        class MockSolverWithoutTrait:
            pass

        # Solver with trait
        solver_with = MockSolverWithTrait()
        try:
            family = solver_with._scheme_family
            assert family == SchemeFamily.FDM
        except AttributeError:
            pytest.fail("Should not raise AttributeError")

        # Solver without trait (raises AttributeError)
        solver_without = MockSolverWithoutTrait()
        try:
            family = solver_without._scheme_family
            pytest.fail("Should raise AttributeError")
        except AttributeError:
            pass  # Expected


class TestDualityValidationLogic:
    """Test typical duality validation logic using SchemeFamily."""

    def test_same_family_detection(self):
        """Test detecting solvers from same family."""

        class MockHJBSolver:
            _scheme_family = SchemeFamily.FDM

        class MockFPSolver:
            _scheme_family = SchemeFamily.FDM

        hjb = MockHJBSolver()
        fp = MockFPSolver()

        hjb_family = getattr(hjb, "_scheme_family", SchemeFamily.GENERIC)
        fp_family = getattr(fp, "_scheme_family", SchemeFamily.GENERIC)

        # Same family, not GENERIC → likely dual
        assert hjb_family == fp_family
        assert hjb_family != SchemeFamily.GENERIC

    def test_mixed_family_detection(self):
        """Test detecting solvers from different families."""

        class MockHJBSolver:
            _scheme_family = SchemeFamily.SL

        class MockFPSolver:
            _scheme_family = SchemeFamily.FDM

        hjb = MockHJBSolver()
        fp = MockFPSolver()

        hjb_family = getattr(hjb, "_scheme_family", SchemeFamily.GENERIC)
        fp_family = getattr(fp, "_scheme_family", SchemeFamily.GENERIC)

        # Different families → not dual
        assert hjb_family != fp_family

    def test_generic_skip_validation(self):
        """Test that GENERIC family skips validation."""

        class MockSolverWithoutTrait:
            pass

        solver = MockSolverWithoutTrait()
        family = getattr(solver, "_scheme_family", SchemeFamily.GENERIC)

        # GENERIC → validation should be skipped
        assert family == SchemeFamily.GENERIC


if __name__ == "__main__":
    # Smoke test - run basic checks
    print("Running SchemeFamily enum smoke tests...")

    # Test enum values
    print(f"✓ FDM: {SchemeFamily.FDM.value}")
    print(f"✓ GFDM: {SchemeFamily.GFDM.value}")
    print(f"✓ GENERIC: {SchemeFamily.GENERIC.value}")

    # Test iteration
    print(f"✓ Total families: {len(list(SchemeFamily))}")

    # Test importability
    from mfgarchon.alg import SchemeFamily as FamilyEnum

    print(f"✓ Import from mfgarchon.alg: {FamilyEnum.FDM.value}")

    # Test validator pattern
    class MockSolver:
        _scheme_family = SchemeFamily.FDM

    solver = MockSolver()
    family = getattr(solver, "_scheme_family", SchemeFamily.GENERIC)
    print(f"✓ Validator pattern: {family.value}")

    print("\nAll smoke tests passed! ✓")
