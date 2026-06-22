"""
Unit tests for adjoint duality validation (Issue #580).

Tests the check_solver_duality() function and DualityStatus classification.
"""

import pytest

import numpy as np

from mfgarchon.alg import SchemeFamily
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils import (
    DualityStatus,
    DualityValidationResult,
    check_solver_duality,
    validate_scheme_config,
)


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(),
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


class TestDualityStatus:
    """Test DualityStatus enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert DualityStatus.DISCRETE_DUAL.value == "discrete_dual"
        assert DualityStatus.CONTINUOUS_DUAL.value == "continuous_dual"
        assert DualityStatus.NOT_DUAL.value == "not_dual"
        assert DualityStatus.VALIDATION_SKIPPED.value == "validation_skipped"

    def test_enum_membership(self):
        """Test enum membership."""
        assert DualityStatus.DISCRETE_DUAL in DualityStatus
        assert DualityStatus.CONTINUOUS_DUAL in DualityStatus
        assert DualityStatus.NOT_DUAL in DualityStatus
        assert DualityStatus.VALIDATION_SKIPPED in DualityStatus


class TestDualityValidationResult:
    """Test DualityValidationResult class."""

    def test_initialization(self):
        """Test result object initialization."""
        result = DualityValidationResult(
            status=DualityStatus.DISCRETE_DUAL,
            hjb_family=SchemeFamily.FDM,
            fp_family=SchemeFamily.FDM,
            message="Test message",
            recommendation="Test recommendation",
        )

        assert result.status == DualityStatus.DISCRETE_DUAL
        assert result.hjb_family == SchemeFamily.FDM
        assert result.fp_family == SchemeFamily.FDM
        assert result.message == "Test message"
        assert result.recommendation == "Test recommendation"

    def test_is_valid_pairing_discrete(self):
        """Test is_valid_pairing() for discrete duality."""
        result = DualityValidationResult(
            status=DualityStatus.DISCRETE_DUAL,
            hjb_family=SchemeFamily.FDM,
            fp_family=SchemeFamily.FDM,
            message="Valid",
        )
        assert result.is_valid_pairing()

    def test_is_valid_pairing_continuous(self):
        """Test is_valid_pairing() for continuous duality."""
        result = DualityValidationResult(
            status=DualityStatus.CONTINUOUS_DUAL,
            hjb_family=SchemeFamily.GFDM,
            fp_family=SchemeFamily.GFDM,
            message="Valid",
        )
        assert result.is_valid_pairing()

    def test_is_valid_pairing_not_dual(self):
        """Test is_valid_pairing() for non-dual pairs."""
        result = DualityValidationResult(
            status=DualityStatus.NOT_DUAL,
            hjb_family=SchemeFamily.FDM,
            fp_family=SchemeFamily.GFDM,
            message="Not dual",
        )
        assert not result.is_valid_pairing()

    def test_requires_renormalization(self):
        """Test requires_renormalization() flag."""
        # Continuous duality requires renormalization
        result_continuous = DualityValidationResult(
            status=DualityStatus.CONTINUOUS_DUAL,
            hjb_family=SchemeFamily.GFDM,
            fp_family=SchemeFamily.GFDM,
            message="Valid",
        )
        assert result_continuous.requires_renormalization()

        # Discrete duality does not require renormalization
        result_discrete = DualityValidationResult(
            status=DualityStatus.DISCRETE_DUAL,
            hjb_family=SchemeFamily.FDM,
            fp_family=SchemeFamily.FDM,
            message="Valid",
        )
        assert not result_discrete.requires_renormalization()

    def test_str_representation(self):
        """Test __str__ method."""
        result = DualityValidationResult(
            status=DualityStatus.DISCRETE_DUAL,
            hjb_family=SchemeFamily.FDM,
            fp_family=SchemeFamily.FDM,
            message="Valid discrete dual",
            recommendation=None,
        )
        str_repr = str(result)
        assert "discrete_dual" in str_repr
        assert "fdm" in str_repr
        assert "Valid discrete dual" in str_repr


class TestCheckSolverDualityFDM:
    """Test duality validation for FDM solvers."""

    def test_fdm_hjb_fp_match(self):
        """Test that FDM HJB and FP solvers are discrete dual."""
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        result = check_solver_duality(HJBFDMSolver, FPFDMSolver)

        assert result.status == DualityStatus.DISCRETE_DUAL
        assert result.hjb_family == SchemeFamily.FDM
        assert result.fp_family == SchemeFamily.FDM
        assert result.is_valid_pairing()
        assert not result.requires_renormalization()

    def test_fdm_weno_compatibility(self):
        """Test that WENO and FDM are compatible (both FDM family)."""
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBWENOSolver

        result = check_solver_duality(HJBWENOSolver, FPFDMSolver)

        assert result.status == DualityStatus.DISCRETE_DUAL
        assert result.hjb_family == SchemeFamily.FDM
        assert result.fp_family == SchemeFamily.FDM


class TestCheckSolverDualitySL:
    """Test duality validation for Semi-Lagrangian solvers."""

    def test_sl_hjb_fp_match(self):
        """Test that SL HJB and FP solvers are discrete dual."""
        from mfgarchon.alg.numerical.fp_solvers import FPSLSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

        result = check_solver_duality(HJBSemiLagrangianSolver, FPSLSolver)

        assert result.status == DualityStatus.DISCRETE_DUAL
        assert result.hjb_family == SchemeFamily.SL
        assert result.fp_family == SchemeFamily.SL
        assert result.is_valid_pairing()
        assert not result.requires_renormalization()

    def test_sl_adjoint_match(self):
        """Test that SL HJB pairs with SL Adjoint FP (forward splatting)."""
        from mfgarchon.alg.numerical.fp_solvers import FPSLAdjointSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

        result = check_solver_duality(HJBSemiLagrangianSolver, FPSLAdjointSolver)

        assert result.status == DualityStatus.DISCRETE_DUAL
        assert result.hjb_family == SchemeFamily.SL
        assert result.fp_family == SchemeFamily.SL


class TestCheckSolverDualityGFDM:
    """Test duality validation for GFDM solvers."""

    def test_gfdm_hjb_fp_match(self):
        """Test that GFDM HJB and FP solvers have continuous duality."""
        from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

        result = check_solver_duality(HJBGFDMSolver, FPGFDMSolver)

        assert result.status == DualityStatus.CONTINUOUS_DUAL
        assert result.hjb_family == SchemeFamily.GFDM
        assert result.fp_family == SchemeFamily.GFDM
        assert result.is_valid_pairing()
        assert result.requires_renormalization()  # Type B needs renorm


class TestCheckSolverDualityMixed:
    """Test duality validation for mixed (non-dual) pairs."""

    def test_fdm_hjb_gfdm_fp_mismatch(self):
        """Test that FDM HJB with GFDM FP is not dual."""
        from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        result = check_solver_duality(HJBFDMSolver, FPGFDMSolver, warn_on_mismatch=False)

        assert result.status == DualityStatus.NOT_DUAL
        assert result.hjb_family == SchemeFamily.FDM
        assert result.fp_family == SchemeFamily.GFDM
        assert not result.is_valid_pairing()

    def test_sl_hjb_fdm_fp_mismatch(self):
        """Test that SL HJB with FDM FP is not dual."""
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

        result = check_solver_duality(HJBSemiLagrangianSolver, FPFDMSolver, warn_on_mismatch=False)

        assert result.status == DualityStatus.NOT_DUAL
        assert result.hjb_family == SchemeFamily.SL
        assert result.fp_family == SchemeFamily.FDM
        assert not result.is_valid_pairing()

    def test_mismatch_warning_emitted(self):
        """Test that mismatched pairs emit warnings when warn_on_mismatch=True."""
        from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        with pytest.warns(UserWarning, match="DUALITY MISMATCH WARNING"):
            result = check_solver_duality(HJBFDMSolver, FPGFDMSolver, warn_on_mismatch=True)

        assert result.status == DualityStatus.NOT_DUAL

    def test_mismatch_warning_suppressed(self):
        """Test that warnings can be suppressed with warn_on_mismatch=False."""
        from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        # Should not emit warning
        result = check_solver_duality(HJBFDMSolver, FPGFDMSolver, warn_on_mismatch=False)
        assert result.status == DualityStatus.NOT_DUAL


class TestCheckSolverDualityGeneric:
    """Test duality validation with GENERIC scheme family."""

    def test_particle_solver_skips_validation(self):
        """Test that GENERIC family solvers skip validation."""
        from mfgarchon.alg.numerical.fp_solvers import FPParticleSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        result = check_solver_duality(HJBFDMSolver, FPParticleSolver)

        assert result.status == DualityStatus.VALIDATION_SKIPPED
        assert result.hjb_family == SchemeFamily.FDM
        assert result.fp_family == SchemeFamily.GENERIC


class TestCheckSolverDualityUnannotated:
    """Test duality validation with unannotated solvers."""

    def test_missing_trait_skips_validation(self):
        """Test that solvers without _scheme_family skip validation."""

        class MockHJBSolver:
            """Mock solver without _scheme_family trait."""

        class MockFPSolver:
            """Mock solver without _scheme_family trait."""

        result = check_solver_duality(MockHJBSolver, MockFPSolver)

        assert result.status == DualityStatus.VALIDATION_SKIPPED
        assert result.hjb_family is None
        assert result.fp_family is None
        assert "missing _scheme_family trait" in result.message


class TestCheckSolverDualityInstances:
    """Test that check_solver_duality works with both classes and instances."""

    def test_validation_with_instances(self):
        """Test validation works when passing solver instances."""
        from mfgarchon import MFGProblem
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        # Create a minimal problem for initialization
        geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[11], boundary_conditions=no_flux_bc(dimension=1))
        problem = MFGProblem(geometry=geometry, Nt=5, T=1.0, components=_default_components())

        hjb_instance = HJBFDMSolver(problem)
        fp_instance = FPFDMSolver(problem)

        result = check_solver_duality(hjb_instance, fp_instance)

        assert result.status == DualityStatus.DISCRETE_DUAL
        assert result.hjb_family == SchemeFamily.FDM
        assert result.fp_family == SchemeFamily.FDM


class TestValidateSchemeConfig:
    """Test validate_scheme_config() for Safe Mode integration."""

    def test_matching_scheme_and_solvers(self):
        """Test that matching scheme and solvers validate successfully."""
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
        from mfgarchon.types import NumericalScheme

        result = validate_scheme_config(NumericalScheme.FDM_UPWIND, HJBFDMSolver, FPFDMSolver)

        assert result.status == DualityStatus.DISCRETE_DUAL
        assert result.is_valid_pairing()

    def test_mismatched_scheme_and_solvers(self):
        """Test that mismatched scheme and solvers are detected."""
        from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
        from mfgarchon.types import NumericalScheme

        result = validate_scheme_config(NumericalScheme.FDM_UPWIND, HJBFDMSolver, FPGFDMSolver)

        assert result.status == DualityStatus.NOT_DUAL
        assert "expects matching families" in result.message

    def test_string_scheme_conversion(self):
        """Test that string scheme names are converted to enum."""
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        result = validate_scheme_config("FDM_UPWIND", HJBFDMSolver, FPFDMSolver)

        assert result.status == DualityStatus.DISCRETE_DUAL

    def test_invalid_scheme_string(self):
        """Test that invalid scheme strings are handled."""
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        result = validate_scheme_config("INVALID_SCHEME", HJBFDMSolver, FPFDMSolver)

        assert result.status == DualityStatus.VALIDATION_SKIPPED
        assert "Unknown scheme" in result.message


class TestDualityValidationEdgeCases:
    """Test edge cases and error handling."""

    def test_none_solver_classes(self):
        """Test handling of None solver classes (missing _scheme_family)."""

        class MockHJBWithTrait:
            _scheme_family = SchemeFamily.FDM

        # Passing None returns VALIDATION_SKIPPED (NoneType has no _scheme_family)
        result = check_solver_duality(MockHJBWithTrait, None)
        assert result.status == DualityStatus.VALIDATION_SKIPPED
        assert result.hjb_family == SchemeFamily.FDM
        assert result.fp_family is None

    def test_same_class_both_sides(self):
        """Test using same solver class for both HJB and FP."""
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        # Should still validate (though nonsensical MFG setup)
        result = check_solver_duality(HJBFDMSolver, HJBFDMSolver)

        assert result.status == DualityStatus.DISCRETE_DUAL
        assert result.hjb_family == SchemeFamily.FDM
        assert result.fp_family == SchemeFamily.FDM


if __name__ == "__main__":
    # Smoke test - run basic checks
    print("Running adjoint validation smoke tests...")

    from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver, FPGFDMSolver
    from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver, HJBGFDMSolver

    # Test FDM pairing (should be discrete dual)
    result = check_solver_duality(HJBFDMSolver, FPFDMSolver)
    assert result.status == DualityStatus.DISCRETE_DUAL
    print(f"✓ FDM pairing: {result.status.value}")

    # Test GFDM pairing (should be continuous dual)
    result = check_solver_duality(HJBGFDMSolver, FPGFDMSolver)
    assert result.status == DualityStatus.CONTINUOUS_DUAL
    assert result.requires_renormalization()
    print(f"✓ GFDM pairing: {result.status.value} (needs renorm)")

    # Test mixed pairing (should be not dual)
    result = check_solver_duality(HJBFDMSolver, FPGFDMSolver, warn_on_mismatch=False)
    assert result.status == DualityStatus.NOT_DUAL
    print(f"✓ Mixed pairing: {result.status.value}")

    print("\nAll smoke tests passed! ✓")
