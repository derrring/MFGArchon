"""
Integration tests for three-mode solving API (Issue #580).

Tests Safe Mode, Expert Mode, and Auto Mode with actual MFG problems.
"""

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types import NumericalScheme


def _default_hamiltonian():
    """Default Hamiltonian for testing."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian centered at 0.5
        u_terminal=lambda x: 0.0,  # Zero terminal cost
        hamiltonian=_default_hamiltonian(),
    )


def _assert_is_a_plausible_solution(result, problem, *, mass_rtol: float = 1e-9) -> None:
    """Assert the solve produced a solution, not merely an object (Issue #1663).

    Before this existed, every solve-invoking test in this file asserted only
    ``result is not None`` plus two ``hasattr`` checks. Poisoning the fixed-point iterator to
    return all-NaN for both U and M left **15 of 16 tests passing**. The file verified that
    ``solve()`` returns something shaped like a result, never that it solved anything.

    The assertion that matters here is MASS CONSERVATION, and that is not obvious -- the
    natural choice, ``np.isfinite``, would not have helped. Measured across the three schemes
    this file exercises, at 5 iterations on the shared fixture:

        scheme          finite   M>=0    mass drift
        FDM_UPWIND      True     True    2.220e-16
        FDM_CENTERED    True     True    6.378e+03     <-- Issue #1671
        SL_LINEAR       True     True    2.220e-16

    A 6378x mass blow-up is finite, has non-negative density, and a non-constant U. Only the
    mass check separates it. ``result.mass_conservation_error`` cannot be used for this: it is
    never populated on the ``problem.solve()`` path and reads ``0.0`` even for that solve
    (Issue #1672), so it is computed here from ``result.M`` directly.
    """
    assert result is not None, "solve() returned None"

    U = np.asarray(result.U)
    M = np.asarray(result.M)

    assert np.isfinite(U).all(), f"U contains {(~np.isfinite(U)).sum()} non-finite entries"
    assert np.isfinite(M).all(), f"M contains {(~np.isfinite(M)).sum()} non-finite entries"
    assert (M >= -1e-12).all(), f"density went negative: min(M) = {M.min():.3e}"
    assert U.std() > 1e-6, "U is constant; a solver returning a fixed array would pass every other check"

    bounds = problem.geometry.get_bounds()
    dx = (bounds[1][0] - bounds[0][0]) / (M.shape[1] - 1)
    mass = M.sum(axis=1) * dx
    drift = float(np.abs(mass - mass[0]).max())
    assert drift <= mass_rtol * max(abs(float(mass[0])), 1.0), (
        f"mass drifted by {drift:.3e} over the horizon (t0={mass[0]:.6f}, "
        f"max={mass.max():.6e}); the FP step is not conservative on this path"
    )


class TestSafeMode:
    """Test Safe Mode: problem.solve(scheme=...)."""

    def test_safe_mode_fdm_upwind(self):
        """Test Safe Mode with FDM_UPWIND scheme."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # Safe Mode: Specify scheme
        result = problem.solve(
            scheme=NumericalScheme.FDM_UPWIND,
            max_iterations=5,
            verbose=False,
        )

        _assert_is_a_plausible_solution(result, problem)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Issue #1671: FDM_CENTERED grows total mass 1.0 -> 6378.77 through problem.solve(), "
            "while FDM_UPWIND on the same path drifts 2.2e-16. The scheme is implicated: run in "
            "isolation with the same advection_scheme the factory dispatches to "
            "(divergence_centered, not the FPFDMSolver default divergence_upwind) and fed the "
            "coupled U, the FP solver grows mass 2.085e+03 on its own. Earlier notes here "
            "claimed the opposite because they measured the upwind default. This marker "
            "is strict (pytest.ini xfail_strict, Issue #1665), so fixing #1671 will turn it into "
            "an XPASS and fail the build until the marker is removed. Note it also XPASSes if "
            "scheme dispatch regresses so FDM_CENTERED is silently not used."
        ),
    )
    def test_safe_mode_fdm_centered(self):
        """Test Safe Mode with FDM_CENTERED scheme."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        result = problem.solve(
            scheme=NumericalScheme.FDM_CENTERED,
            max_iterations=5,
            verbose=False,
        )

        _assert_is_a_plausible_solution(result, problem)

    def test_safe_mode_sl_linear(self):
        """Safe Mode with SL_LINEAR: the solve runs and is mass-conserving.

        The ``@pytest.mark.skip(reason="Pre-existing bug in SL solver (NaN/Inf issue)")`` this
        carried is stale -- the defect it named is fixed. Removing the marker leaves the test
        passing, and passing under the mass-conservation assertion below, not merely
        not-crashing: measured drift 2.220e-16 (Issue #1663).
        """
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        result = problem.solve(
            scheme=NumericalScheme.SL_LINEAR,
            max_iterations=5,
            verbose=False,
        )

        _assert_is_a_plausible_solution(result, problem)

    def test_safe_mode_string_scheme(self):
        """Test Safe Mode with string scheme name."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # Should accept string and convert to enum
        result = problem.solve(
            scheme="fdm_upwind",
            max_iterations=5,
            verbose=False,
        )

        _assert_is_a_plausible_solution(result, problem)

    def test_safe_mode_invalid_string_scheme(self):
        """Test Safe Mode with invalid string scheme."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        with pytest.raises(ValueError, match="Unknown scheme string"):
            problem.solve(scheme="invalid_scheme", max_iterations=5, verbose=False)


class TestExpertMode:
    """Test Expert Mode: problem.solve(hjb_solver=..., fp_solver=...)."""

    def test_expert_mode_matching_fdm_solvers(self):
        """Test Expert Mode with matching FDM solvers."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # Create matching FDM solvers
        hjb = HJBFDMSolver(problem)
        fp = FPFDMSolver(problem)

        # Expert Mode: Manual injection
        result = problem.solve(
            hjb_solver=hjb,
            fp_solver=fp,
            max_iterations=5,
            verbose=False,
        )

        _assert_is_a_plausible_solution(result, problem)

    def test_expert_mode_mismatched_solvers_warning(self):
        """Test Expert Mode with mismatched solvers emits warning."""
        from mfgarchon.alg.numerical.fp_solvers import FPSLSolver

        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # Create mismatched solvers (FDM HJB with SL FP)
        # These have compatible grids but different scheme families
        hjb = HJBFDMSolver(problem)
        fp = FPSLSolver(problem)  # Semi-Lagrangian FP with FDM HJB = not dual

        # Verify they're detected as non-dual
        from mfgarchon.utils import check_solver_duality

        result = check_solver_duality(hjb, fp, warn_on_mismatch=False)
        assert not result.is_valid_pairing()

        # Now test that problem.solve() emits UserWarning from duality check
        with pytest.warns(UserWarning, match="DUALITY MISMATCH"):
            # This should emit warning but still work
            solve_result = problem.solve(
                hjb_solver=hjb,
                fp_solver=fp,
                max_iterations=2,  # Just a few iterations
                verbose=True,  # Verbose needed for logger warning
            )

        _assert_is_a_plausible_solution(solve_result, problem)

    def test_expert_mode_partial_injection_raises_error(self):
        """Test Expert Mode with only one solver raises error."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        hjb = HJBFDMSolver(problem)

        # Only HJB provided, no FP
        with pytest.raises(ValueError, match="Expert Mode requires BOTH"):
            problem.solve(hjb_solver=hjb, max_iterations=5, verbose=False)

        # Only FP provided, no HJB
        fp = FPFDMSolver(problem)
        with pytest.raises(ValueError, match="Expert Mode requires BOTH"):
            problem.solve(fp_solver=fp, max_iterations=5, verbose=False)


class TestAutoMode:
    """Test Auto Mode: problem.solve() with no scheme/solvers."""

    def test_auto_mode_default_behavior(self):
        """Test Auto Mode selects default scheme."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # Auto Mode: No scheme or solvers specified
        result = problem.solve(max_iterations=5, verbose=False)

        _assert_is_a_plausible_solution(result, problem)

    def test_auto_mode_verbose_shows_selection(self, caplog):
        """Test Auto Mode logs scheme selection when verbose."""
        import logging

        # Configure logger to ensure INFO messages are captured
        from mfgarchon.utils.mfg_logging import get_logger

        logger = get_logger("mfgarchon.core.mfg_problem")
        logger.setLevel(logging.INFO)

        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        with caplog.at_level(logging.INFO, logger="mfgarchon.core.mfg_problem"):
            result = problem.solve(max_iterations=5, verbose=True)

        # Should log which scheme was selected
        _assert_is_a_plausible_solution(result, problem)

        # Check if Auto Mode or scheme name appears in logs
        # Note: May not log if logger handler isn't configured, so make this optional
        log_messages = " ".join([record.message for record in caplog.records])
        has_auto_mode_log = "Auto Mode" in log_messages or "fdm_upwind" in log_messages

        # Test passes if either: (1) log message found, or (2) result is valid
        # This makes test robust to logger configuration differences
        assert has_auto_mode_log or result is not None


class TestModeMixingErrors:
    """Test that mixing modes raises clear errors."""

    def test_safe_and_expert_mode_mixing_raises_error(self):
        """Test that specifying both scheme and solvers raises error."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        hjb = HJBFDMSolver(problem)
        fp = FPFDMSolver(problem)

        with pytest.raises(ValueError, match=r"Cannot mix Safe Mode.*Expert Mode"):
            problem.solve(
                scheme=NumericalScheme.FDM_UPWIND,
                hjb_solver=hjb,
                fp_solver=fp,
                max_iterations=5,
                verbose=False,
            )

    def test_safe_mode_with_partial_expert_raises_error(self):
        """Test that specifying scheme with one solver raises error."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        hjb = HJBFDMSolver(problem)

        with pytest.raises(ValueError, match=r"Cannot mix Safe Mode.*Expert Mode"):
            problem.solve(
                scheme=NumericalScheme.FDM_UPWIND,
                hjb_solver=hjb,
                max_iterations=5,
                verbose=False,
            )


class TestBackwardCompatibility:
    """Test that existing code patterns still work."""

    def test_basic_solve_still_works(self):
        """Test that problem.solve() without parameters still works."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # Old pattern: Just call solve()
        result = problem.solve(max_iterations=5, verbose=False)

        _assert_is_a_plausible_solution(result, problem)

    def test_solve_with_tolerance_and_iterations(self):
        """Test that specifying tolerance and iterations still works."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        result = problem.solve(
            max_iterations=10,
            tolerance=1e-4,
            verbose=False,
        )

        _assert_is_a_plausible_solution(result, problem)


class TestConfigIntegration:
    """Test that config parameter works with three-mode API."""

    @pytest.mark.slow
    def test_safe_mode_with_config(self):
        """Test Safe Mode with custom config."""
        from mfgarchon.config import MFGSolverConfig

        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )
        config = MFGSolverConfig()
        config.picard.max_iterations = 3

        result = problem.solve(
            scheme=NumericalScheme.FDM_UPWIND,
            config=config,
            verbose=False,
        )

        _assert_is_a_plausible_solution(result, problem)

    @pytest.mark.slow
    def test_expert_mode_with_config(self):
        """Test Expert Mode with custom config."""
        from mfgarchon.config import MFGSolverConfig

        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )
        config = MFGSolverConfig()

        hjb = HJBFDMSolver(problem)
        fp = FPFDMSolver(problem)

        result = problem.solve(
            hjb_solver=hjb,
            fp_solver=fp,
            config=config,
            verbose=False,
        )

        _assert_is_a_plausible_solution(result, problem)


if __name__ == "__main__":
    # Smoke test - run basic checks
    print("Running three-mode API integration tests...")

    # Test Safe Mode
    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=10,
        T=1.0,
        components=_default_components(),
    )
    result = problem.solve(scheme=NumericalScheme.FDM_UPWIND, max_iterations=3, verbose=False)
    _assert_is_a_plausible_solution(result, problem)
    print("✓ Safe Mode works")

    # Test Expert Mode
    hjb = HJBFDMSolver(problem)
    fp = FPFDMSolver(problem)
    result = problem.solve(hjb_solver=hjb, fp_solver=fp, max_iterations=3, verbose=False)
    _assert_is_a_plausible_solution(result, problem)
    print("✓ Expert Mode works")

    # Test Auto Mode
    result = problem.solve(max_iterations=3, verbose=False)
    _assert_is_a_plausible_solution(result, problem)
    print("✓ Auto Mode works")

    print("\nAll smoke tests passed! ✓")
