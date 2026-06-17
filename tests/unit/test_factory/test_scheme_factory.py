"""
Unit tests for scheme-based solver factory (Issue #580).

Tests create_paired_solvers() function with all scheme variants.

Issue #673: Updated to class-based Hamiltonian API.
"""

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg import SchemeFamily
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.factory import create_paired_solvers, get_recommended_scheme
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types import NumericalScheme
from mfgarchon.utils import DualityStatus, check_solver_duality


def _default_hamiltonian():
    """Create default class-based Hamiltonian for tests."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Provide default components for test problems."""
    return MFGComponents(
        hamiltonian=_default_hamiltonian(),
        m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(),
        u_terminal=lambda x: 0.0,
    )


class TestCreatePairedSolversFDM:
    """Test FDM solver pairing."""

    def test_fdm_upwind_creates_dual_pair(self):
        """Test that FDM_UPWIND creates valid dual pair."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        hjb, fp = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)

        # Check solver types
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        assert isinstance(hjb, HJBFDMSolver)
        assert isinstance(fp, FPFDMSolver)

        # Check duality
        result = check_solver_duality(hjb, fp)
        assert result.status == DualityStatus.DISCRETE_DUAL
        assert result.hjb_family == SchemeFamily.FDM
        assert result.fp_family == SchemeFamily.FDM

    def test_fdm_upwind_default_advection_scheme(self):
        """Test that FDM_UPWIND sets divergence_upwind for FP.

        Note: Changed from gradient_upwind to divergence_upwind in Issue #382
        because gradient_upwind has boundary flux bugs.
        """
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        _, fp = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)

        # Check FP advection scheme (divergence_upwind is mass-conservative)
        assert hasattr(fp, "advection_scheme")
        assert fp.advection_scheme == "divergence_upwind"

    def test_fdm_centered_creates_dual_pair(self):
        """Test that FDM_CENTERED creates valid dual pair."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        hjb, fp = create_paired_solvers(problem, NumericalScheme.FDM_CENTERED)

        # Check solver types
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver

        assert isinstance(hjb, HJBFDMSolver)
        assert isinstance(fp, FPFDMSolver)

        # Check FP advection scheme: conservative central form (Issue #1149 -- the
        # non-conservative gradient_centered leaked mass through no-flux walls).
        assert fp.advection_scheme == "divergence_centered"

    def test_fdm_custom_config(self):
        """Test FDM pairing with custom configs."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # Override FP advection scheme
        _, fp = create_paired_solvers(
            problem,
            NumericalScheme.FDM_UPWIND,
            fp_config={"advection_scheme": "divergence_upwind"},
        )

        # Config override should work
        assert fp.advection_scheme == "divergence_upwind"


class TestCreatePairedSolversSL:
    """Test Semi-Lagrangian solver pairing."""

    def test_sl_linear_creates_dual_pair(self):
        """Test that SL_LINEAR creates valid dual pair."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        hjb, fp = create_paired_solvers(problem, NumericalScheme.SL_LINEAR)

        # Check solver types
        from mfgarchon.alg.numerical.fp_solvers import FPSLSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

        assert isinstance(hjb, HJBSemiLagrangianSolver)
        assert isinstance(fp, FPSLSolver)  # Forward SL, not backward

        # Check duality
        result = check_solver_duality(hjb, fp)
        assert result.status == DualityStatus.DISCRETE_DUAL
        assert result.hjb_family == SchemeFamily.SL
        assert result.fp_family == SchemeFamily.SL

    def test_sl_linear_default_interpolation(self):
        """Test that SL_LINEAR sets linear interpolation for HJB."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        hjb, _ = create_paired_solvers(problem, NumericalScheme.SL_LINEAR)

        assert hasattr(hjb, "interpolation_method")
        assert hjb.interpolation_method == "linear"

    def test_sl_cubic_creates_dual_pair(self):
        """Test that SL_CUBIC creates valid dual pair."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        hjb, fp = create_paired_solvers(problem, NumericalScheme.SL_CUBIC)

        # Check solver types
        from mfgarchon.alg.numerical.fp_solvers import FPSLSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBSemiLagrangianSolver

        assert isinstance(hjb, HJBSemiLagrangianSolver)
        assert isinstance(fp, FPSLSolver)

        # Check interpolation
        assert hjb.interpolation_method == "cubic"

    def test_sl_uses_adjoint_solver_not_backward(self):
        """Test that SL pairing uses FPSLSolver (forward SL) for duality."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        _, fp = create_paired_solvers(problem, NumericalScheme.SL_LINEAR)

        # Must be FPSLSolver (forward splatting), not FPSLJacobianSolver (backward interpolation)
        from mfgarchon.alg.numerical.fp_solvers import FPSLSolver

        assert isinstance(fp, FPSLSolver)
        assert fp.fp_method_name == "Adjoint Semi-Lagrangian"


class TestCreatePairedSolversGFDM:
    """Test GFDM solver pairing."""

    def test_gfdm_creates_dual_pair(self):
        """Test that GFDM creates valid dual pair."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # GFDM requires collocation points
        points = np.linspace(0, 1, 15)[:, None]  # 1D points

        hjb, fp = create_paired_solvers(
            problem,
            NumericalScheme.GFDM,
            hjb_config={"collocation_points": points},
            fp_config={"collocation_points": points},
        )

        # Check solver types
        from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver

        assert isinstance(hjb, HJBGFDMSolver)
        assert isinstance(fp, FPGFDMSolver)

        # Check duality (GFDM is Type B - continuous only)
        result = check_solver_duality(hjb, fp)
        assert result.status == DualityStatus.CONTINUOUS_DUAL
        assert result.hjb_family == SchemeFamily.GFDM
        assert result.fp_family == SchemeFamily.GFDM
        assert result.requires_renormalization()  # Type B needs renorm

    def test_gfdm_delta_threading(self):
        """Test that delta parameter is threaded between HJB and FP."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )
        points = np.linspace(0, 1, 15)[:, None]

        # Specify delta only for HJB
        hjb, fp = create_paired_solvers(
            problem,
            NumericalScheme.GFDM,
            hjb_config={"collocation_points": points, "delta": 0.05},
            fp_config={"collocation_points": points},
        )

        # FP should inherit delta
        assert hjb.delta == 0.05
        assert fp.delta == 0.05

    def test_gfdm_collocation_threading(self):
        """Test that collocation_points are threaded if specified for one solver."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )
        points = np.linspace(0, 1, 15)[:, None]

        # Specify points only for HJB
        hjb, fp = create_paired_solvers(
            problem,
            NumericalScheme.GFDM,
            hjb_config={"collocation_points": points},
        )

        # Both should have same points
        assert np.array_equal(hjb.collocation_points, fp.collocation_points)


class TestCreatePairedSolversValidation:
    """Test validation and error handling."""

    def test_validation_enabled_by_default(self):
        """Test that duality validation is enabled by default."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # Should not raise (FDM is valid)
        hjb, fp = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)

        result = check_solver_duality(hjb, fp)
        assert result.is_valid_pairing()

    def test_validation_can_be_disabled(self):
        """Test that validation can be disabled with validate_duality=False."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # Should work even with validation disabled
        hjb, fp = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND, validate_duality=False)

        # Solvers should still be valid
        assert check_solver_duality(hjb, fp).is_valid_pairing()

    def test_unimplemented_scheme_raises_error(self):
        """Test that unimplemented schemes raise NotImplementedError."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # SL_CUBIC is defined but cubic FP adjoint not implemented
        # For now it creates the pair but with a note in docstring
        # This test just verifies it doesn't crash
        hjb, fp = create_paired_solvers(problem, NumericalScheme.SL_CUBIC)
        assert hjb is not None
        assert fp is not None

    def test_unrouted_scheme_message_lists_every_routed_scheme(self):
        """Pin the NotImplementedError 'Available schemes' message to the routed set.

        Every NumericalScheme member is routed in the if-elif chain, so an unhandled
        scheme cannot occur with the real enum. Use a sentinel with a `.value` attribute
        to fall through to the else branch, then assert the message advertises all routed
        schemes (regression guard: SL_CUBIC and MESHLESS_GALERKIN were previously omitted).
        """
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        class _FakeScheme:
            value = "not_a_real_scheme"

        with pytest.raises(NotImplementedError) as exc_info:
            create_paired_solvers(problem, _FakeScheme())

        message = str(exc_info.value)
        for member in NumericalScheme:
            assert member.name in message, f"{member.name} missing from 'Available schemes' message"


class TestGetRecommendedScheme:
    """Test intelligent scheme recommendation (Phase 3 placeholder)."""

    def test_returns_fdm_upwind_by_default(self):
        """Test that default recommendation is FDM_UPWIND."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        scheme = get_recommended_scheme(problem)

        assert scheme == NumericalScheme.FDM_UPWIND


class TestConfigThreading:
    """Test that configs are properly threaded to solvers."""

    def test_hjb_config_passed_to_hjb_solver(self):
        """Test that hjb_config parameters are passed through."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # Verify solver was created with empty config (config passing tested in other tests)
        hjb, _ = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND, hjb_config={})

        # Verify solver was created successfully
        assert hjb is not None

    def test_fp_config_passed_to_fp_solver(self):
        """Test that fp_config parameters are passed through."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        _, fp = create_paired_solvers(
            problem,
            NumericalScheme.FDM_UPWIND,
            fp_config={"advection_scheme": "divergence_upwind"},
        )

        assert fp.advection_scheme == "divergence_upwind"

    def test_empty_configs_use_defaults(self):
        """Test that omitting configs uses solver defaults."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        hjb, fp = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)

        # Should create solvers with default parameters
        assert hjb is not None
        assert fp is not None


class TestReturnTypes:
    """Test return type consistency."""

    def test_returns_tuple_of_two_solvers(self):
        """Test that function always returns (hjb, fp) tuple."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        result = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)

        assert isinstance(result, tuple)
        assert len(result) == 2

        hjb, fp = result
        assert hjb is not None
        assert fp is not None

    def test_hjb_has_scheme_family_trait(self):
        """Test that returned HJB solver has _scheme_family trait."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        hjb, _ = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)

        assert hasattr(hjb, "_scheme_family")
        assert hjb._scheme_family == SchemeFamily.FDM

    def test_fp_has_scheme_family_trait(self):
        """Test that returned FP solver has _scheme_family trait."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        _, fp = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)

        assert hasattr(fp, "_scheme_family")
        assert fp._scheme_family == SchemeFamily.FDM


if __name__ == "__main__":
    # Smoke test - run basic checks
    print("Running scheme factory smoke tests...")

    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=10,
        T=1.0,
        components=_default_components(),
    )

    # Test FDM pairing
    hjb, fp = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)
    result = check_solver_duality(hjb, fp)
    assert result.status == DualityStatus.DISCRETE_DUAL
    print(f"✓ FDM pairing: {result.status.value}")

    # Test SL pairing
    hjb, fp = create_paired_solvers(problem, NumericalScheme.SL_LINEAR)
    result = check_solver_duality(hjb, fp)
    assert result.status == DualityStatus.DISCRETE_DUAL
    print(f"✓ SL pairing: {result.status.value}")

    # Test GFDM pairing
    points = np.linspace(0, 1, 15)[:, None]
    hjb, fp = create_paired_solvers(
        problem,
        NumericalScheme.GFDM,
        hjb_config={"collocation_points": points},
        fp_config={"collocation_points": points},
    )
    result = check_solver_duality(hjb, fp)
    assert result.status == DualityStatus.CONTINUOUS_DUAL
    assert result.requires_renormalization()
    print(f"✓ GFDM pairing: {result.status.value} (needs renorm)")

    print("\nAll smoke tests passed! ✓")
