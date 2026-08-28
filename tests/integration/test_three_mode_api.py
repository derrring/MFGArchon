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


def _assert_is_a_plausible_solution(result, problem, *, mass_rtol: float = 1e-9, nodal_sum: bool = False) -> None:
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

    # #2145: the geometry owns the measure. `sum(m)*dx` gives the two wall nodes a full cell each
    # on an endpoint-inclusive grid and is a different functional -- the one the FP wall used to
    # conserve while losing real mass.
    mass = np.asarray(problem.geometry.integrate(M), dtype=float)
    drift = float(np.abs(mass - mass[0]).max())
    assert drift <= mass_rtol * max(abs(float(mass[0])), 1.0), (
        f"mass drifted by {drift:.3e} over the horizon (t0={mass[0]:.6f}, "
        f"max={mass.max():.6e}); the FP step is not conservative on this path"
    )
    if nodal_sum:
        # The invariant a SPLATTING scheme actually has (#2145). `FPSLSolver` is the forward
        # semi-Lagrangian: it traces x_dest = x + alpha*dt and scatters each node's mass to the
        # surrounding nodes with linear weights that sum to 1, so `sum_i m_i` is conserved BY
        # CONSTRUCTION. That is the counting measure. It equals the mass only on a mesh of equal
        # control volumes, and this grid is endpoint-inclusive, so the two wall nodes own half a
        # cell and moving mass on or off them changes the real mass without changing the sum.
        #
        # Asserting it keeps this path honest: a genuine leak still fails here, which a loosened
        # `mass_rtol` alone would not catch.
        nodal = np.asarray(M, dtype=float).reshape(M.shape[0], -1).sum(axis=1)
        nodal_drift = float(np.abs(nodal / nodal[0] - 1.0).max())
        assert nodal_drift < 1e-9, f"the splatting scheme's own invariant broke: nodal-sum drift {nodal_drift:.3e}"


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

        # `FPSLSolver` conserves the NODAL SUM, not the mass (#2145), for TWO reasons, and an
        # earlier version of this note gave only the first.
        #
        # 1. Splatting. The forward semi-Lagrangian scatters each node's mass to its neighbours
        #    with weights summing to 1, so `sum_i m_i` is exact by construction -- the counting
        #    measure, which equals the mass only on a mesh of equal control volumes. This grid is
        #    endpoint-inclusive, so the wall nodes own half a cell and moving mass onto them
        #    changes the mass without changing the sum.
        # 2. **Its diffusion.** `fp_semi_lagrangian_adjoint.py:437-465` hand-writes the zero-flux
        #    Crank-Nicolson wall as `L[0] = (m[1]-m[0])/dx^2` -- the old half stencil, a fifth copy
        #    of the wall the rest of #2145 fixed. Independent review measured it with ZERO
        #    velocity, so splatting displaces nothing: rectangle drift 1.8e-14, trapezoid drift
        #    8.56e-03. So the diffusion alone accounts for most of what this test sees, and the
        #    splat-only account was wrong.
        #
        # RECORDED, not accepted. `nodal_sum=True` pins the invariant the scheme does have, so a
        # genuine leak still reddens this: verified by injecting `m_star * (1 - 1e-6)` into
        # `splat_linear_1d`, which turns both SL sites red; the floor is 3e-10 per step.
        # Retirement needs BOTH halves -- splat `w_i * m_i` and divide by `w_j` on arrival, AND
        # give that Crank-Nicolson wall row the h/2 control volume. Fixing one is measurably worse
        # than fixing neither, which is why this is recorded rather than half-done. Giving only the
        # diffusion its control volume, on a transported fixture:
        #
        #     as shipped              rectangle 8.882e-16 (exact)   trapezoid 6.361e-02
        #     diffusion half fixed    rectangle 7.662e-02           trapezoid 6.033e-03
        #
        # so the scheme goes from conserving one functional exactly to conserving none. (At ZERO
        # velocity the splat displaces nothing and the half-fix does conserve, 3.997e-15 -- which is
        # why the regime has to be named before that measurement means anything.) Both are changes
        # to the scheme, not redirects, so neither is made here.
        # This site: safe mode, SL_LINEAR, 5 Picard iterations. Measured mass drift 7.283e-03
        # against a nodal drift of 2.220e-16.
        _assert_is_a_plausible_solution(result, problem, mass_rtol=2e-2, nodal_sum=True)

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

        # `FPSLSolver` conserves the NODAL SUM, not the mass (#2145), for TWO reasons, and an
        # earlier version of this note gave only the first.
        #
        # 1. Splatting. The forward semi-Lagrangian scatters each node's mass to its neighbours
        #    with weights summing to 1, so `sum_i m_i` is exact by construction -- the counting
        #    measure, which equals the mass only on a mesh of equal control volumes. This grid is
        #    endpoint-inclusive, so the wall nodes own half a cell and moving mass onto them
        #    changes the mass without changing the sum.
        # 2. **Its diffusion.** `fp_semi_lagrangian_adjoint.py:437-465` hand-writes the zero-flux
        #    Crank-Nicolson wall as `L[0] = (m[1]-m[0])/dx^2` -- the old half stencil, a fifth copy
        #    of the wall the rest of #2145 fixed. Independent review measured it with ZERO
        #    velocity, so splatting displaces nothing: rectangle drift 1.8e-14, trapezoid drift
        #    8.56e-03. So the diffusion alone accounts for most of what this test sees, and the
        #    splat-only account was wrong.
        #
        # RECORDED, not accepted. `nodal_sum=True` pins the invariant the scheme does have, so a
        # genuine leak still reddens this: verified by injecting `m_star * (1 - 1e-6)` into
        # `splat_linear_1d`, which turns both SL sites red; the floor is 3e-10 per step.
        # Retirement needs BOTH halves -- splat `w_i * m_i` and divide by `w_j` on arrival, AND
        # give that Crank-Nicolson wall row the h/2 control volume. Fixing one is measurably worse
        # than fixing neither, which is why this is recorded rather than half-done. Giving only the
        # diffusion its control volume, on a transported fixture:
        #
        #     as shipped              rectangle 8.882e-16 (exact)   trapezoid 6.361e-02
        #     diffusion half fixed    rectangle 7.662e-02           trapezoid 6.033e-03
        #
        # so the scheme goes from conserving one functional exactly to conserving none. (At ZERO
        # velocity the splat displaces nothing and the half-fix does conserve, 3.997e-15 -- which is
        # why the regime has to be named before that measurement means anything.) Both are changes
        # to the scheme, not redirects, so neither is made here.
        # This site is NOT the safe-mode one and its number is thirty times larger: expert mode
        # pairs a NON-DUAL `HJBFDMSolver` with `FPSLSolver` for 2 iterations, and the trapezoid mass
        # falls 1.000000 -> 0.776655 in a single step and stays there -- 2.2335e-01, against a nodal
        # drift of 2.220e-16. That is what the bound below is sized for, and an earlier version of
        # this comment pasted the safe-mode paragraph here verbatim, so the site carried numbers
        # from a different solve and nothing explained its tolerance.
        #
        # 22% of the real mass is not a rounding artefact and is not accepted as one: it is the
        # recorded defect above, seen at full size because a non-dual pairing transports harder.
        # `nodal_sum=True` is what keeps this honest -- the invariant the scheme does have is pinned
        # to 1e-9, so a genuine leak still reddens the test even at this tolerance.
        _assert_is_a_plausible_solution(solve_result, problem, mass_rtol=3e-1, nodal_sum=True)

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

    def test_auto_mode_verbose_shows_selection(self, mfg_caplog):
        """Auto Mode names the scheme it selected, on the logger, at INFO.

        The assertion used to be `has_auto_mode_log or result is not None`, whose right branch
        is unconditionally true -- written to be "robust to logger configuration differences"
        when the real difference was that whether `caplog` sees an mfgarchon record depends on
        the pytest version and on when the logger was created (#2083). With `mfg_caplog` the log
        is observable either way, so the assertion can be the log.
        """
        import logging

        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        with mfg_caplog.at_level(logging.INFO, logger="mfgarchon.core.mfg_problem"):
            result = problem.solve(max_iterations=5, verbose=True)

        _assert_is_a_plausible_solution(result, problem)

        log_messages = " ".join(mfg_caplog.messages)
        assert "Auto Mode" in log_messages, (
            f"Auto Mode did not announce the scheme it selected; captured: {mfg_caplog.messages}"
        )
        assert "fdm_upwind" in log_messages, (
            f"the announcement must name the scheme, not only the mode; captured: {mfg_caplog.messages}"
        )


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
