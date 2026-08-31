"""Two reported statuses now describe the run they name (#1684 items 3 and 5).

Both are the same shape as items 1, 2, 6 and 7: a field called `converged` computed from
something other than whether the solve converged. Neither raised; both simply reported success.

**Item 3** -- `DistributionConvergenceMonitor.get_convergence_summary()["converged"]` was
`len(converged_iterations) > 0`, i.e. "some iterate, ever, met the criteria". A run that met them
and then diverged reported success. It now reports the LAST iterate's state, and nothing is lost:
`convergence_iteration` still records the first iterate that met the criteria, so "did it ever
converge" is still answerable and is now a different question from "did it end converged".

**Item 5** -- `MultiPopulationIterator` measured convergence from `M` alone. `U_old` was never
captured, so the value function -- half of the coupled unknown -- could not enter the test at any
tolerance. #1914 records a Picard loop where one field settles three orders while the other
diverges two, which is exactly the run this could not see.

WHAT THESE CATCH, measured by restoring the pre-fix source: **4 of 6 fail, 2 of them
behaviourally.** `test_a_run_that_diverged_after_converging_is_not_converged` and
`test_a_static_m_with_a_moving_u_does_not_converge` each construct a run the old code called
converged and the true answer calls not; both die on that assertion. The other two die on
`AttributeError` for the new breakdown fields, which is honest but proves only that a field was
added -- they are guards on the breakdown, not on the criterion.

That distinction cost a revision worth recording. `test_a_static_m_with_a_moving_u_does_not_converge`
first asserted `errors_M` before `converged`, so against the old code it died on the missing
attribute and never evaluated the question it exists to ask. A test can fail against the broken
code for the wrong reason and look like discrimination. The behavioural assertion is first now.

The two passing tests pin what must NOT be lost: a genuinely converged run still reports True, or
the fix would be a constant `False` and equally decoupled from the solve.
"""

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfgarchon.core import MFGComponents, MFGProblem
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc
from mfgarchon.utils.convergence.convergence_monitors import DistributionConvergenceMonitor

# --- item 3 ------------------------------------------------------------------------------------
#
# These drive `convergence_history` directly. That is white box, and deliberate: it is the exact
# mechanism #1684 item 3 describes, and reaching the same state through `update()` would require
# manufacturing a wasserstein/stabilisation sequence, which would test the criteria rather than the
# summary.


def _monitor_with(history_flags):
    m = DistributionConvergenceMonitor()
    for i, flag in enumerate(history_flags):
        m.convergence_history.append({"iteration": i + 1, "u_l2_error": 1e-12 if flag else 5e19, "converged": flag})
    return m


def test_a_run_that_diverged_after_converging_is_not_converged():
    """The counterfactual. The old code reported True here."""
    summary = _monitor_with([False, True, False, False]).get_convergence_summary()
    assert summary["converged"] is False, (
        f"a run whose last iterate has u error {summary['final_u_error']:.1e} reported converged"
    )
    assert summary["convergence_iteration"] == 1, (
        "the first converged iterate must still be recorded; 'did it ever converge' is a real "
        "question and this fix must not delete the answer"
    )


def test_a_run_that_ended_converged_is_converged():
    """Guards against the fix being a constant False."""
    summary = _monitor_with([False, False, True]).get_convergence_summary()
    assert summary["converged"] is True
    assert summary["convergence_iteration"] == 2


def test_a_run_that_never_converged_is_not_converged():
    summary = _monitor_with([False, False]).get_convergence_summary()
    assert summary["converged"] is False
    assert summary["convergence_iteration"] is None


# --- item 5 ------------------------------------------------------------------------------------

_NX, _NT, _T, _SIG = 12, 4, 1.0, 0.15


def _problem():
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: np.zeros_like(np.asarray(m, float)),
        coupling_dm=lambda m: np.zeros_like(np.asarray(m, float)),
        population_index=0,
    )
    comps = MFGComponents(
        m_initial=lambda xx: np.exp(-((np.asarray(xx) - 0.5) ** 2) / 0.02),
        u_terminal=lambda xx: np.asarray(xx) * 0.0,
        hamiltonian=H,
    )
    return MFGProblem(
        geometry=TensorProductGrid(
            bounds=[(0.0, 1.0)], Nx_points=[_NX + 1], boundary_conditions=no_flux_bc(dimension=1)
        ),
        Nt=_NT,
        T=_T,
        sigma=_SIG,
        components=comps,
    )


class _StaticFP(FPFDMSolver):
    """Returns the same density every sweep, so `max|dM|` is exactly 0 from sweep 2 on."""

    def solve_fp_system(self, m_initial, *args, source_term=None, volatility_field=None, **kwargs):
        # `source_term` / `volatility_field` are named because the base class refuses a bare
        # **kwargs that would swallow them (#2020). This stub honours neither, and the fixture
        # supplies neither, so silence is correct here rather than a quiet drop.
        assert source_term is None, "this stub cannot honour a source term"
        assert volatility_field is None, "this stub cannot honour a volatility field"
        return np.tile(np.asarray(m_initial, float), (_NT + 1, 1))


class _DriftingHJB(HJBFDMSolver):
    """Returns a value function that moves by exactly 1.0 every sweep and never settles."""

    def __init__(self, problem):
        super().__init__(problem)
        self._sweep = 0

    def solve_hjb_system(self, *args, source_term=None, volatility_field=None, **kwargs):
        # Named for the same reason as the FP stub above (#2020): a bare **kwargs would swallow
        # them. Neither is supplied by this fixture and neither is honoured.
        assert source_term is None, "this stub cannot honour a source term"
        assert volatility_field is None, "this stub cannot honour a volatility field"
        self._sweep += 1
        return np.full((_NT + 1, _NX + 1), float(self._sweep))


def _run(max_iterations=3):
    p = _problem()
    multi = MultiPopulationProblem(populations=[p], population_names=["P0"])
    it = MultiPopulationIterator(multi, [_DriftingHJB(p)], [_StaticFP(p)], relaxation=1.0)
    return it.solve(max_iterations=max_iterations, tolerance=1e-10)


def test_a_static_m_with_a_moving_u_does_not_converge():
    """The counterfactual. m is bit-identical between sweeps while u moves by 1.0 each time.

    The old criterion saw only m and reported convergence at the second sweep.
    """
    r = _run()
    # The behavioural assertion comes FIRST, on purpose. Asserting the new `errors_M` /
    # `errors_U` fields first made this test fail against the old code with AttributeError --
    # proving only that a field had been added, and never reaching the question the test exists
    # to ask. Ordered this way it fails on the criterion being blind to u, which is the defect.
    assert r.converged is False, (
        f"reported converged with a tolerance of 1e-10 while the value function was still moving; "
        f"per-population error reported as {r.errors}"
    )
    # Fixture validity, and the breakdown that says which field failed.
    assert r.errors_M == pytest.approx([0.0], abs=1e-14), f"the fixture is wrong if m moved: errors_M={r.errors_M}"
    assert r.errors_U[0] == pytest.approx(1.0), f"u should move by 1.0 per sweep: {r.errors_U}"


def test_the_m_error_is_the_map_residual_not_the_damped_step():
    """#1684 items 6 and 7, in the multi-population path. The counterfactual is the relaxation.

    `M = (1 - r) * M_old + r * M_map`, so `M - M_old` is identically `r * (M_map - M_old)`. An
    error measured on the damped update therefore shrinks with the relaxation factor and turning
    damping down makes anything converge -- which is what items 6 and 7 name, and what
    `test_fixed_point_residual_is_undamped_1684.py` and
    `test_coupling_metric_is_the_map_residual_1684.py` already pin for the single-population path.

    Measured on this fixture, one sweep, before the fix: err_M was 1.785179e-01 / 8.925895e-02 /
    1.785179e-02 / 1.785179e-03 at relaxation 1.0 / 0.5 / 0.1 / 0.01 -- ratios 1.00 / 2.00 / 10.00 /
    100.00, exactly 1/r. After: 1.785179e-01 at every one of them.

    This asserts INVARIANCE rather than a value, so it survives any change to the fixture or the
    scheme that does not reintroduce the damping dependence.
    """
    errs = []
    for relaxation in (1.0, 0.5, 0.1, 0.01):
        p = _problem()
        multi = MultiPopulationProblem(populations=[p], population_names=["P0"])
        it = MultiPopulationIterator(multi, [HJBFDMSolver(p)], [FPFDMSolver(p)], relaxation=relaxation)
        errs.append(it.solve(max_iterations=1, tolerance=1e-30).errors_M[0])

    assert errs[0] > 0, "the fixture must actually move m, or this asserts nothing"
    for r, e in zip((1.0, 0.5, 0.1, 0.01), errs, strict=True):
        assert e == pytest.approx(errs[0], rel=1e-12), (
            f"err_M moved with the relaxation factor (r={r}: {e:.6e} vs {errs[0]:.6e}, ratio "
            f"{errs[0] / e:.2f}). The convergence test is measuring the damped step, so lowering "
            f"the damping buys the verdict -- #1684 items 6 and 7."
        )


def test_the_reported_error_is_the_larger_of_the_two_fields():
    r = _run()
    assert r.errors == [max(m, u) for m, u in zip(r.errors_M, r.errors_U, strict=True)]


def test_the_split_says_which_field_failed():
    """A non-converged run must say WHICH field failed, not only that one did."""
    r = _run()
    assert r.errors_U > r.errors_M, (
        f"u is the field that failed here and the breakdown should show it: "
        f"errors_U={r.errors_U}, errors_M={r.errors_M}"
    )
