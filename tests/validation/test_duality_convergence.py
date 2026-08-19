"""
Convergence validation for adjoint duality (Issue #580).

This test verifies that dual solver pairs produce better convergence
than non-dual pairs, validating the mathematical correctness of the
adjoint pairing system.

Mathematical Theory:
-------------------
For dual schemes (Type A), the discrete operators satisfy:
    L_FP = L_HJB^T exactly

This ensures that the Nash gap converges at optimal rate:
    Nash_gap = O(h^2) for second-order schemes

For non-dual pairs, the transpose relationship is broken:
    L_FP ≠ L_HJB^T

This leads to persistent Nash gap:
    Nash_gap = O(1) even as h → 0

References:
-----------
- Issue #580: Adjoint-aware solver pairing
- Issue #706 (adjoint operators)
- Issue #580 (adjoint pairing implementation)
"""

import pytest

import numpy as np

from mfgarchon import MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.types import NumericalScheme


def _default_hamiltonian():
    """Default class-based Hamiltonian for tests (Issue #673)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        hamiltonian=_default_hamiltonian(),
        m_initial=lambda x: np.exp(-10 * (np.asarray(x) - 0.5) ** 2).squeeze(),
        u_terminal=lambda x: 0.0,
    )


class TestDualityConvergence:
    """Test that dual pairs converge better than non-dual pairs."""

    @pytest.mark.slow
    def test_dual_fdm_pair_converges(self):
        """Test that FDM dual pair achieves good convergence."""
        # Create problem with known solution characteristics
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[40 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=20,
            T=1.0,
            sigma=0.1,
            components=_default_components(),
        )

        # Solve with dual FDM pair (Safe Mode)
        result = problem.solve(
            scheme=NumericalScheme.FDM_UPWIND,
            max_iterations=50,
            tolerance=1e-8,
            verbose=False,
        )

        # Check convergence quality
        assert result.converged or result.iterations >= 30, "FDM dual pair should converge or make progress"

        # Check that errors show overall progress (not necessarily monotonic)
        # MFG Picard iteration can oscillate, especially early iterations
        errors = np.array(result.error_history_U[: min(10, len(result.error_history_U))])
        if len(errors) > 3:
            # Final error should be smaller than max of first 3 errors (overall progress)
            initial_max = np.max(errors[:3])
            final_error = errors[-1]
            assert final_error <= initial_max * 2.0, "Should show overall error reduction trend"

    # test_centered_fdm_higher_order was removed 2026-07-25 rather than repaired.
    #
    # It ran ONE grid (Nx=41), so it could not measure an order under any assertion, despite
    # the name. Its assertion was `result.converged or result.iterations >= 30` against
    # max_iterations=50 -- a non-converging solve runs to 50, so the disjunction held either
    # way and the test could not fail on its own terms. And its configuration was byte-identical
    # to TestNumericalStability::test_centered_fdm_may_oscillate below, which asserts that this
    # exact problem MUST raise: at sigma=0.1 on this grid the cell Peclet number puts
    # divergence_centered in its oscillatory regime (Issue #1671). Two tests, one configuration,
    # opposite contracts; the one with the evidence is the one that stayed.
    #
    # Not replaced with a real EOC test here: a probe over the existing coupled MMS at low
    # Peclet gave error ratios 1.84 / 1.72 across a two-grid sequence, which is not a stable
    # second-order signal, so writing one would have meant choosing a threshold to fit the
    # numbers. Genuine centered-scheme order verification belongs with the MMS convergence
    # suite, not as a single-grid smoke test named after a rate.

    # test_mesh_refinement_improves_accuracy was removed 2026-08-19 rather than repaired. It is
    # what #1998 -- "Nightly full test suite is failing" -- actually was: the validation shard
    # reports `1 failed, 27 passed, 4 xfailed`, that one test, every other shard green.
    #
    # It recorded `result.max_error`, which is the final PICARD RESIDUAL, not a discretization
    # error against a known solution, and compared two of them across Nx = 20 / 40 under the name
    # "h-convergence". Measured: with tolerance=1e-8 and max_iterations=30 BOTH legs run to the cap
    # -- `converged=False, iterations=30/30` on either grid. At max_iterations=300 neither converges
    # either. The full history, both legs:
    #
    #     Nx=20  peak 1.264638e+03 @iter4   best 1.778986e-06 @iter194   @iter29 5.935364e-04
    #     Nx=40  peak 1.243941e+03 @iter4   best 7.116063e-06 @iter166   @iter29 4.237936e-04
    #
    # A transient peaking above 1.2e+03 around iteration 4, then a settled 1e-4..2e-3 noise floor
    # it never leaves -- final value far from its own best. So `max_error` at iteration 30 is a
    # sample off that floor, and no assertion over two samples can measure a rate.
    #
    # The precedent is TestConvergenceRate::test_upwind_first_order_convergence, removed for this
    # exact reason with its account at the bottom of this file -- "It measured the wrong quantity
    # ... it compared `result.max_error` ... the final PICARD RESIDUAL". #1728 names that test. (The
    # removal noted ABOVE, test_centered_fdm_higher_order, is a different defect: one grid and a
    # tautological assertion. Same file, same cleanup, not the same reason.)
    #
    # WHAT THIS IS NOT: a clean window. A real numerical change landed inside it. c98a9c5f
    # ("stop overwriting the root the inner Newton just found", #1902) deletes the hand-rolled
    # `u[0] = u[1] - g*dx` no-flux enforcement -- the exact boundary treatment this test's
    # `no_flux_bc(dimension=1)` runs through. It moves these numbers on darwin -- ~5x on the coarse
    # leg, 1.31x DOWNWARD on the fine one -- so the quantity that moves as a single number is the
    # ratio the assertion tests: coarse/fine goes 0.2136 -> 1.4005, a factor 6.56:
    #
    #     179e55a7 (head of the last GREEN nightly)  Nx=20 1.185395e-04  Nx=40 5.550650e-04  FAILS
    #     c98a9c5f and after                         Nx=20 5.935364e-04  Nx=40 4.237936e-04  passes
    #     linux (CI), 4a50e27b and after             Nx=20 8.814973e-05  Nx=40 9.901234e-04  FAILS
    #
    # On darwin the flip bisects to c98a9c5f. On linux only the ENDPOINTS are known -- green at
    # 179e55a7, red at 4a50e27b, sixteen commits between -- but c98a9c5f is the only commit in the
    # window that moves this quantity on the platform we can measure, and the obvious rival
    # (2e7a909a, the Neumann ghost consolidation) is bracketed by two bit-identical commits.
    # So the verdict flipped in OPPOSITE directions on the two platforms,
    # which is what a limit-cycling residual does and is stronger grounds for deletion than a
    # clean-window story would have been. No platform conditional is involved, and CI pins the BLAS
    # thread counts to 1. Anyone tracing an FDM_UPWIND no-flux discrepancy to this window should
    # start at #1902 and #1904, not conclude the window is computationally quiet.
    #
    # What the deletion costs. `e < 1000` was NOT a vacuous bound -- the residual crosses it, in
    # this very solve, by 26%: the peak above is 1.26e+03. So that assertion was a crude but LIVE
    # guard on the TRANSIENT DECAY RATE. Anything costing roughly eight Picard iterations of
    # convergence speed pushes iteration 30 back over 1000 and fires it. Nothing else in the suite
    # bounds that, and no claim of domination survives: `max_error = max(err_U, err_M)` is here
    # entirely the U increment (at Nx=40, iteration 29: err_U 4.238e-04 against err_M 4.807e-08),
    # while test_fdm_upwind_stable's mass-conservation and `M.min() > 0` assertions constrain M's
    # physics. The two sets are incomparable, not ordered. That test also stops at iteration 20.
    #
    # The rest of the cost is smaller than it first looked. The Nx=40 leg duplicates
    # test_fdm_upwind_stable on every listed parameter -- Nx_points=[41], Nt=20, T=1.0, sigma=0.1,
    # same components, FDM_UPWIND -- differing only in max_iterations (30 vs 20) and tolerance, so
    # iterations 21-30 there are unexercised. And the COARSE leg is not uncovered in kind:
    # tests/integration/test_fvm_hjb_coupling.py's `fdm_1d` fixture runs a coupled 1D no-flux
    # FDM_UPWIND solve at Nx=25, Nt=12, sigma=0.4 and asserts convergence, a 10x error drop, mass
    # and positivity -- strictly stronger, at comparable coarseness. What has no counterpart is the
    # LOW-SIGMA / high-Peclet coarse regime, and the transient guard above.
    #
    # Deleting is still right: a threshold on an unconverged residual is a poor instrument even
    # when live, and the RATE assertion beside it is indefensible at any threshold. But the
    # transient goes unguarded, and that is recorded rather than claimed away.
    #
    # Coupled EOC through the production FixedPointIterator already exists:
    # tests/integration/test_coupled_mfg_mms.py::TestCoupledMMSConvergence. What remains uncovered
    # is coupled EOC AT A NO-FLUX WALL -- that suite is periodic -- and the deleted test never
    # measured it either. The standalone FP order study added by #2006
    # (tests/unit/test_alg/test_fp_mms_wall_order_1728.py) covers the wall but not the coupling.

    def test_safe_mode_guarantees_duality(self):
        """Test that Safe Mode automatically creates dual pairs."""
        from mfgarchon.factory import create_paired_solvers
        from mfgarchon.utils import check_solver_duality

        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # Create pair via Safe Mode factory
        hjb, fp = create_paired_solvers(
            problem,
            NumericalScheme.FDM_UPWIND,
            validate_duality=True,
        )

        # Verify duality
        result = check_solver_duality(hjb, fp, warn_on_mismatch=False)
        assert result.is_valid_pairing(), "Safe Mode should guarantee duality"
        assert result.status.value in ["discrete_dual", "continuous_dual"]

    def test_expert_mode_detects_mismatch(self):
        """Test that Expert Mode detects non-dual pairs."""
        from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver, FPSLSolver
        from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
        from mfgarchon.utils import check_solver_duality

        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=10,
            T=1.0,
            components=_default_components(),
        )

        # Create dual pair
        hjb_fdm = HJBFDMSolver(problem)
        fp_fdm = FPFDMSolver(problem)

        result_dual = check_solver_duality(hjb_fdm, fp_fdm, warn_on_mismatch=False)
        assert result_dual.is_valid_pairing(), "FDM-FDM should be dual"

        # Create non-dual pair
        fp_sl = FPSLSolver(problem)

        result_nondual = check_solver_duality(hjb_fdm, fp_sl, warn_on_mismatch=False)
        assert not result_nondual.is_valid_pairing(), "FDM-SL should not be dual"


class TestConvergenceRate:
    """Empty since 2026-07-25 -- see the note below for why nothing replaced its one test.

    Kept rather than deleted so the note stays attached to the class it explains.
    """

    # test_upwind_first_order_convergence was removed 2026-07-25. It measured the wrong
    # quantity, and upwind's spatial order is verified properly elsewhere.
    #
    # It refined the grid and compared `result.max_error`, which is
    # `max(final_error_U, final_error_M)` (solver_result.py:169-171) -- the final PICARD
    # RESIDUAL, i.e. how much the last iteration moved, not a discretization error against
    # a known solution. That quantity has no reason to scale like h. It scales with how hard
    # the coupled iteration is at that resolution, and refining raises the cell Peclet number,
    # so the residual after a fixed 30-iteration budget grows: measured ratio 3.636, which is
    # what made this test fail. No amount of threshold tuning turns a residual into an order.
    #
    # Two further defects made it unable to fail for the right reason either: the assertion sat
    # behind `if all(e < 10.0 for e in errors)`, so a badly-behaved run skipped the check
    # silently, and the accepted band `0.1 < ratio <= 1.5` admits a 50% error INCREASE under
    # refinement while being named after first-order convergence.
    #
    # NOT replaced, and upwind's spatial order is now UNCOVERED. The first version of this
    # deletion cited TestMMSFokkerPlanck1D::test_sinusoidal_periodic_convergence as the real
    # coverage. Review refuted that: it solves with `potential_field=U_zero`
    # (test_mms_validation.py:260), i.e. zero drift, so the advection stencil never runs.
    # Substituting divergence_centered or gradient_upwind for divergence_upwind leaves its
    # errors bit-identical -- it measures the diffusion/BC treatment, not the scheme its ratios
    # are attributed to. The coverage delta of this deletion is still zero (the removed test
    # could not measure an order either), but nothing in this repo now fails if upwind stops
    # being first-order. Tracked in #1728.


@pytest.mark.slow
class TestNumericalStability:
    """Test that dual schemes maintain stability."""

    def test_fdm_upwind_stable(self):
        """Test that upwind FDM is stable (monotone)."""
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[40 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=20,
            T=1.0,
            sigma=0.1,
            components=_default_components(),
        )

        result = problem.solve(
            scheme=NumericalScheme.FDM_UPWIND,
            max_iterations=20,
            verbose=False,
        )

        # Check for NaN or Inf (indicates instability)
        assert np.all(np.isfinite(result.U)), "U should remain finite (stable)"
        assert np.all(np.isfinite(result.M)), "M should remain finite (stable)"

        # Check that density stays positive
        assert np.all(result.M >= -1e-10), "Density should remain non-negative"

        # Both checks above are properties the REPAIR supplies rather than the physics -- the
        # point test_centered_fdm_may_oscillate below records from the other side: isfinite plus
        # a non-negativity bound passed there while the clip fabricated 0.013% of the mass per
        # firing, and "only mass can" distinguish that from a healthy solve. So assert mass.
        M = np.asarray(result.M)
        dx = 1.0 / 40
        mass = M.sum(axis=1) * dx
        # Measured max|mass - mass[0]| = 2.7e-15 (mass[0] = 1.0 exactly); 1e-12 is a ~375x margin.
        assert np.max(np.abs(mass - mass[0])) < 1e-12, "no-flux upwind FP must conserve mass"

        # Strict positivity, which is the discriminating form: clip_nonnegative_or_raise returns
        # np.maximum(density, 0.0), so a clipped entry is exactly 0.0 and the `>= -1e-10` bound
        # above cannot see it, while `> 0` can. Measured min M = 6.4e-07 with zero entries equal
        # to 0.0, i.e. the clip never fired on this configuration.
        assert M.min() > 0.0, "a strictly positive density means the non-negativity clip never fired"

    def test_centered_fdm_may_oscillate(self):
        """Centered FDM oscillates here, and the FP solver now refuses rather than repairing it.

        This asserted only ``np.all(np.isfinite(...))`` and passed, because the non-negativity
        clip silently raised every negative density to zero -- fabricating 0.013% of the total
        mass per firing at this configuration, while leaving the output finite and non-negative
        (Issue #1671). ``isfinite`` cannot distinguish that from a healthy solve; only mass can.

        With the guard in place the same configuration raises, so the test now pins the refusal.
        That is the intended behaviour: at sigma=0.1 on this grid the cell Peclet number puts
        ``divergence_centered`` in its oscillatory regime, and the honest outcome is a diagnostic,
        not a plausible-looking answer.
        """
        problem = MFGProblem(
            geometry=TensorProductGrid(
                bounds=[(0.0, 1.0)], Nx_points=[40 + 1], boundary_conditions=no_flux_bc(dimension=1)
            ),
            Nt=20,
            T=1.0,
            sigma=0.1,
            components=_default_components(),
        )

        with pytest.raises(ValueError, match="would fabricate"):
            problem.solve(
                scheme=NumericalScheme.FDM_CENTERED,
                max_iterations=20,
                verbose=False,
            )


if __name__ == "__main__":
    # Smoke test - run quick validation
    print("Running duality convergence validation...\n")

    from mfgarchon.factory import create_paired_solvers
    from mfgarchon.utils import check_solver_duality

    # Test 1: Verify dual pairing
    print("Test 1: Safe Mode duality guarantee")
    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[20 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=10,
        T=1.0,
        components=_default_components(),
    )
    hjb, fp = create_paired_solvers(problem, NumericalScheme.FDM_UPWIND)
    result = check_solver_duality(hjb, fp)
    assert result.is_valid_pairing()
    print(f"  ✓ Status: {result.status.value}")
    print(f"  ✓ HJB family: {result.hjb_family.value}")
    print(f"  ✓ FP family: {result.fp_family.value}")

    # Test 2: Verify convergence
    print("\nTest 2: Convergence with dual pair")
    problem = MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[40 + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=20,
        T=1.0,
        sigma=0.1,
        components=_default_components(),
    )
    solve_result = problem.solve(
        scheme=NumericalScheme.FDM_UPWIND,
        max_iterations=30,
        verbose=False,
    )
    print(f"  ✓ Converged: {solve_result.converged}")
    print(f"  ✓ Iterations: {solve_result.iterations}")
    print(f"  ✓ Final error: {solve_result.max_error:.3e}")

    # Test 3: Check stability
    print("\nTest 3: Numerical stability")
    assert np.all(np.isfinite(solve_result.U))
    assert np.all(np.isfinite(solve_result.M))
    assert np.all(solve_result.M >= -1e-10)
    print("  ✓ Solution finite")
    print("  ✓ Density non-negative")

    print("\n" + "=" * 50)
    print("All validation tests passed! ✓")
    print("=" * 50)
