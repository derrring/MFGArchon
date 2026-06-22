"""
Unit tests for solver _scheme_family trait annotations.

Tests that all HJB and FP solvers are correctly annotated with _scheme_family
traits for duality validation (Issue #580).
"""

from mfgarchon.alg import SchemeFamily


class TestHJBSolverTraits:
    """Test _scheme_family trait annotations on HJB solvers."""

    def test_hjb_fdm_has_fdm_trait(self):
        """Test that HJBFDMSolver has FDM scheme family."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        assert hasattr(HJBFDMSolver, "_scheme_family")
        assert HJBFDMSolver._scheme_family == SchemeFamily.FDM

    def test_hjb_semi_lagrangian_has_sl_trait(self):
        """Test that HJBSemiLagrangianSolver has SL scheme family."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver

        assert hasattr(HJBSemiLagrangianSolver, "_scheme_family")
        assert HJBSemiLagrangianSolver._scheme_family == SchemeFamily.SL

    def test_hjb_gfdm_has_gfdm_trait(self):
        """Test that HJBGFDMSolver has GFDM scheme family."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver

        assert hasattr(HJBGFDMSolver, "_scheme_family")
        assert HJBGFDMSolver._scheme_family == SchemeFamily.GFDM

    def test_hjb_weno_has_fdm_trait(self):
        """Test that HJBWENOSolver has FDM scheme family (WENO is FDM variant)."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver

        assert hasattr(HJBWENOSolver, "_scheme_family")
        assert HJBWENOSolver._scheme_family == SchemeFamily.FDM

    def test_all_hjb_solvers_have_trait(self):
        """Test that all HJB solvers have _scheme_family trait."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import (
            HJBSemiLagrangianSolver,
        )
        from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver

        hjb_solvers = [
            HJBFDMSolver,
            HJBSemiLagrangianSolver,
            HJBGFDMSolver,
            HJBWENOSolver,
        ]

        for solver_class in hjb_solvers:
            assert hasattr(solver_class, "_scheme_family"), f"{solver_class.__name__} missing _scheme_family"
            assert isinstance(solver_class._scheme_family, SchemeFamily), (
                f"{solver_class.__name__}._scheme_family is not SchemeFamily enum"
            )

    def test_hjb_fdm_variants_both_fdm(self):
        """Test that both FDM and WENO solvers have FDM family."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver

        # Both should be FDM family (WENO is high-order FDM)
        assert HJBFDMSolver._scheme_family == SchemeFamily.FDM
        assert HJBWENOSolver._scheme_family == SchemeFamily.FDM

    def test_hjb_trait_types_are_consistent(self):
        """Test that all HJB solver traits are valid SchemeFamily values."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import (
            HJBSemiLagrangianSolver,
        )
        from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver

        # All should be valid SchemeFamily enum members
        all_families = set(SchemeFamily)

        assert HJBFDMSolver._scheme_family in all_families
        assert HJBSemiLagrangianSolver._scheme_family in all_families
        assert HJBGFDMSolver._scheme_family in all_families
        assert HJBWENOSolver._scheme_family in all_families


class TestValidatorPatternWithTraits:
    """Test using validator pattern (getattr) with solver traits."""

    def test_getattr_pattern_with_hjb_fdm(self):
        """Test getattr pattern for retrieving _scheme_family trait."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        # Direct access
        family = getattr(HJBFDMSolver, "_scheme_family", SchemeFamily.GENERIC)
        assert family == SchemeFamily.FDM
        assert family != SchemeFamily.GENERIC

    def test_getattr_pattern_with_unannotated_class(self):
        """Test getattr pattern falls back to GENERIC for unannotated classes."""

        class MockUnannotatedSolver:
            """Mock solver without _scheme_family trait."""

        family = getattr(MockUnannotatedSolver, "_scheme_family", SchemeFamily.GENERIC)
        assert family == SchemeFamily.GENERIC

    def test_getattr_pattern_for_all_hjb_solvers(self):
        """Test getattr pattern works for all annotated HJB solvers."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import (
            HJBSemiLagrangianSolver,
        )
        from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver

        hjb_solvers = [
            (HJBFDMSolver, SchemeFamily.FDM),
            (HJBSemiLagrangianSolver, SchemeFamily.SL),
            (HJBGFDMSolver, SchemeFamily.GFDM),
            (HJBWENOSolver, SchemeFamily.FDM),
        ]

        for solver_class, expected_family in hjb_solvers:
            family = getattr(solver_class, "_scheme_family", SchemeFamily.GENERIC)
            assert family == expected_family, f"{solver_class.__name__} has wrong family"
            assert family != SchemeFamily.GENERIC, f"{solver_class.__name__} should not be GENERIC"


class TestDualityValidationPreparation:
    """Test that trait annotations enable duality validation logic."""

    def test_same_family_detection_fdm(self):
        """Test detecting when HJB and FP solvers from same FDM family."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        # Mock FP solver with FDM trait (will be implemented in Phase 1.4)
        class MockFPFDMSolver:
            _scheme_family = SchemeFamily.FDM

        hjb_family = getattr(HJBFDMSolver, "_scheme_family", SchemeFamily.GENERIC)
        fp_family = getattr(MockFPFDMSolver, "_scheme_family", SchemeFamily.GENERIC)

        # Same family, not GENERIC → likely dual (Type A)
        assert hjb_family == fp_family
        assert hjb_family != SchemeFamily.GENERIC

    def test_mixed_family_detection(self):
        """Test detecting when HJB and FP solvers from different families."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        # Mock FP solver with different family
        class MockFPSLSolver:
            _scheme_family = SchemeFamily.SL

        hjb_family = getattr(HJBFDMSolver, "_scheme_family", SchemeFamily.GENERIC)
        fp_family = getattr(MockFPSLSolver, "_scheme_family", SchemeFamily.GENERIC)

        # Different families → not dual
        assert hjb_family != fp_family

    def test_discrete_vs_continuous_duality_classification(self):
        """Test classifying schemes by duality type based on family."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import (
            HJBSemiLagrangianSolver,
        )

        # Type A: Discrete duality (exact transpose)
        discrete_duality_families = {SchemeFamily.FDM, SchemeFamily.SL, SchemeFamily.FVM}

        fdm_family = HJBFDMSolver._scheme_family
        sl_family = HJBSemiLagrangianSolver._scheme_family

        assert fdm_family in discrete_duality_families  # Type A
        assert sl_family in discrete_duality_families  # Type A

        # Type B: Continuous duality only (asymptotic transpose)
        continuous_only_families = {SchemeFamily.GFDM, SchemeFamily.PINN}

        gfdm_family = HJBGFDMSolver._scheme_family

        assert gfdm_family in continuous_only_families  # Type B

    def test_duality_validation_logic_simulation(self):
        """Simulate the duality validation logic that will be in Phase 2."""
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver

        def simulate_check_duality(hjb_class, fp_class):
            """Simulate check_solver_duality() logic."""
            hjb_family = getattr(hjb_class, "_scheme_family", SchemeFamily.GENERIC)
            fp_family = getattr(fp_class, "_scheme_family", SchemeFamily.GENERIC)

            # Skip validation if either is GENERIC
            if hjb_family == SchemeFamily.GENERIC or fp_family == SchemeFamily.GENERIC:
                return "validation_skipped"

            # Check if same family
            if hjb_family != fp_family:
                return "not_dual"

            # Classify duality type
            discrete_families = {SchemeFamily.FDM, SchemeFamily.SL, SchemeFamily.FVM}
            if hjb_family in discrete_families:
                return "discrete_dual"  # Type A
            else:
                return "continuous_dual"  # Type B

        # Mock FP solvers
        class MockFPFDM:
            _scheme_family = SchemeFamily.FDM

        class MockFPGFDM:
            _scheme_family = SchemeFamily.GFDM

        class MockFPUnannotated:
            pass

        # Test validation logic
        assert simulate_check_duality(HJBFDMSolver, MockFPFDM) == "discrete_dual"
        assert simulate_check_duality(HJBGFDMSolver, MockFPGFDM) == "continuous_dual"
        assert simulate_check_duality(HJBFDMSolver, MockFPGFDM) == "not_dual"
        assert simulate_check_duality(HJBFDMSolver, MockFPUnannotated) == "validation_skipped"


class TestTraitImportability:
    """Test that traits don't cause circular imports or other issues."""

    def test_hjb_solvers_import_cleanly(self):
        """Test that all HJB solvers can be imported without errors."""
        # Should not raise ImportError
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import (
            HJBSemiLagrangianSolver,
        )
        from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver

        # All should have trait
        assert HJBFDMSolver._scheme_family is not None
        assert HJBSemiLagrangianSolver._scheme_family is not None
        assert HJBGFDMSolver._scheme_family is not None
        assert HJBWENOSolver._scheme_family is not None

    def test_no_circular_import_with_scheme_family(self):
        """Test that importing solvers doesn't cause circular import."""
        # Import SchemeFamily first
        from mfgarchon.alg import SchemeFamily

        # Then import solvers - should not raise
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        # Verify trait matches imported enum
        assert HJBFDMSolver._scheme_family == SchemeFamily.FDM


class TestFPSolverTraits:
    """Test _scheme_family trait annotations on FP solvers."""

    def test_fp_fdm_has_fdm_trait(self):
        """Test that FPFDMSolver has FDM scheme family."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver

        assert hasattr(FPFDMSolver, "_scheme_family")
        assert FPFDMSolver._scheme_family == SchemeFamily.FDM

    def test_fp_sl_has_sl_trait(self):
        """Test that FPSLSolver (forward SL) has SL scheme family."""
        from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver

        assert hasattr(FPSLSolver, "_scheme_family")
        assert FPSLSolver._scheme_family == SchemeFamily.SL

    def test_fp_sl_jacobian_has_sl_trait(self):
        """Test that FPSLJacobianSolver (backward SL, deprecated) has SL scheme family."""
        from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian import FPSLJacobianSolver

        assert hasattr(FPSLJacobianSolver, "_scheme_family")
        assert FPSLJacobianSolver._scheme_family == SchemeFamily.SL

    def test_fp_gfdm_has_gfdm_trait(self):
        """Test that FPGFDMSolver has GFDM scheme family."""
        from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver

        assert hasattr(FPGFDMSolver, "_scheme_family")
        assert FPGFDMSolver._scheme_family == SchemeFamily.GFDM

    def test_fp_particle_has_generic_trait(self):
        """Test that FPParticleSolver has GENERIC scheme family."""
        from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

        assert hasattr(FPParticleSolver, "_scheme_family")
        assert FPParticleSolver._scheme_family == SchemeFamily.GENERIC

    def test_all_fp_solvers_have_trait(self):
        """Test that all FP solvers have _scheme_family trait."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
        from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
        from mfgarchon.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
        from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian import FPSLJacobianSolver
        from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver

        fp_solvers = [
            FPFDMSolver,
            FPSLSolver,
            FPSLJacobianSolver,
            FPGFDMSolver,
            FPParticleSolver,
        ]

        for solver_class in fp_solvers:
            assert hasattr(solver_class, "_scheme_family"), f"{solver_class.__name__} missing _scheme_family"
            assert isinstance(solver_class._scheme_family, SchemeFamily), (
                f"{solver_class.__name__}._scheme_family is not SchemeFamily enum"
            )

    def test_fp_sl_variants_both_sl(self):
        """Test that both forward SL and backward SL (Jacobian) solvers have SL family."""
        from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian import FPSLJacobianSolver
        from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver

        # Both should be SL family (forward splatting and backward Jacobian variants)
        assert FPSLSolver._scheme_family == SchemeFamily.SL
        assert FPSLJacobianSolver._scheme_family == SchemeFamily.SL


class TestHJBFPPairing:
    """Test that HJB and FP solvers have matching traits for duality."""

    def test_fdm_hjb_fp_match(self):
        """Test that FDM HJB and FP solvers have matching families."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        assert HJBFDMSolver._scheme_family == FPFDMSolver._scheme_family
        assert HJBFDMSolver._scheme_family == SchemeFamily.FDM

    def test_sl_hjb_fp_match(self):
        """Test that SL HJB and FP solvers have matching families."""
        from mfgarchon.alg.numerical.fp_solvers.fp_semi_lagrangian_adjoint import FPSLSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import (
            HJBSemiLagrangianSolver,
        )

        assert HJBSemiLagrangianSolver._scheme_family == FPSLSolver._scheme_family
        assert HJBSemiLagrangianSolver._scheme_family == SchemeFamily.SL

    def test_gfdm_hjb_fp_match(self):
        """Test that GFDM HJB and FP solvers have matching families."""
        from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver

        assert HJBGFDMSolver._scheme_family == FPGFDMSolver._scheme_family
        assert HJBGFDMSolver._scheme_family == SchemeFamily.GFDM

    def test_duality_pairing_simulation(self):
        """Simulate the pairing logic that will be in Phase 2 factory."""
        from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
        from mfgarchon.alg.numerical.fp_solvers.fp_gfdm import FPGFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver

        # FDM pairing: Same family, discrete duality
        assert HJBFDMSolver._scheme_family == FPFDMSolver._scheme_family
        assert HJBFDMSolver._scheme_family == SchemeFamily.FDM

        # GFDM pairing: Same family, continuous duality only
        assert HJBGFDMSolver._scheme_family == FPGFDMSolver._scheme_family
        assert HJBGFDMSolver._scheme_family == SchemeFamily.GFDM

        # Mixed pairing: Different families, not dual
        assert HJBFDMSolver._scheme_family != FPGFDMSolver._scheme_family


if __name__ == "__main__":
    # Smoke test - run basic checks
    print("Running solver trait smoke tests...")

    from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_semi_lagrangian import HJBSemiLagrangianSolver
    from mfgarchon.alg.numerical.hjb_solvers.hjb_weno import HJBWENOSolver

    # Test all HJB solvers have traits
    solvers = [
        (HJBFDMSolver, SchemeFamily.FDM, "HJB FDM"),
        (HJBSemiLagrangianSolver, SchemeFamily.SL, "HJB Semi-Lagrangian"),
        (HJBGFDMSolver, SchemeFamily.GFDM, "HJB GFDM"),
        (HJBWENOSolver, SchemeFamily.FDM, "HJB WENO"),
    ]

    for solver_class, expected_family, name in solvers:
        family = getattr(solver_class, "_scheme_family", None)
        assert family == expected_family, f"{name} has wrong family"
        print(f"✓ {name}: {family.value}")

    # Test validator pattern
    family = getattr(HJBFDMSolver, "_scheme_family", SchemeFamily.GENERIC)
    assert family == SchemeFamily.FDM
    print(f"✓ Validator pattern works: {family.value}")

    print("\nAll smoke tests passed! ✓")
