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

    @pytest.mark.slow
    def test_mesh_refinement_improves_accuracy(self):
        """Test that finer meshes reduce errors (h-convergence)."""
        mesh_sizes = [20, 40]
        final_errors = []

        for Nx in mesh_sizes:
            problem = MFGProblem(
                geometry=TensorProductGrid(
                    bounds=[(0.0, 1.0)], Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1)
                ),
                Nt=Nx // 2,
                T=1.0,
                sigma=0.1,
                components=_default_components(),
            )

            result = problem.solve(
                scheme=NumericalScheme.FDM_UPWIND,
                max_iterations=30,
                tolerance=1e-8,
                verbose=False,
            )

            # Record final error (max of U and M errors)
            final_errors.append(result.max_error)

        # Finer mesh should have smaller error
        # We can't guarantee strict convergence in all cases (problem-dependent),
        # but we can check that errors are reasonable
        assert all(e < 1000 for e in final_errors), "Errors should be bounded"

        # If both converged well, expect refinement to help
        if all(e < 1.0 for e in final_errors):
            # Coarse mesh error should be larger (or comparable)
            # Allow some tolerance for numerical noise
            assert final_errors[0] >= final_errors[1] * 0.5, "Refinement should improve or maintain accuracy"

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
