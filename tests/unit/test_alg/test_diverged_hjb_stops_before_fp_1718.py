"""A diverged HJB is attributed to HJB, at every coupling loop (#1718).

#1717 fixed one loop; its pre-merge review found the same shape at six more. Each is
HJB-solve -> FP-consumes-the-result, with nothing looking in between, and each fails the same way:
the FP solver composes a drift from NaN and raises ``"Check CFL condition: dt * sigma^2 / dx^2
should be < 0.5"``. That is an FP diagnostic, carrying advice that would send a reader to shrink a
timestep, for an HJB failure.

Structured after `test_paired_solver_sigma_single_source_1603.py`, which pins the same
"one owner, wired into every iterator" property for the paired-sigma guard. The parallel is not
decorative: that guard was also filed against one iterator, also found live at the others, and the
list iterators were also the gap.

WHAT THESE CATCH, measured against the pre-#1718 tree: **every test in this file fails, and the
call-site tests fail behaviourally.** (That was 10 of 10 when first written; the file has since
grown to 13, so the count is stated as a property rather than a number that rots.) All four call-site tests die inside `_run_to_completion` with
`ValueError: FP solver produced NaN/Inf at timestep 3/5 ... Check CFL condition` -- the exact
misattribution this issue is about, reproduced. The 6 owner tests die on ImportError, which is
honest but proves only that a symbol was added.

That split is the reason the owner is imported inside `_owner()` rather than at module scope. At
module scope the pre-#1718 tree fails during COLLECTION, the call-site tests never run, and the
whole check degrades to "the symbol is missing".

`_run_to_completion` is the load-bearing assertion. A test that only asserted `converged is False`
would pass the old code by accident wherever the exception is swallowed, and error rather than fail
where it is not -- so these assert first that the loop RETURNS, then what it returned.

~~`GraphMFGSolver` is patched by the same commit and is not exercised here: it needs a graph
geometry this fixture cannot build.~~ [SUPERSEDED 2026-08-31] SUPERSEDED-BY: the review of #2194.
**Both halves of that were false and it is retracted here, in the sentence that made it, not only
in the test that refutes it.** There is no graph geometry: `test_graph_mfg_solver.py`'s
`_make_3node_system` builds the solver from three ordinary 1-D problems and a 3x3 adjacency matrix,
the shape `_make_problem` above already makes. And the similarity argument did not hold either --
the regime test read only `values[0]`, so it never covered the part it was being leaned on for.
`test_graph_solver_stops_and_publishes_a_usable_result` below now covers that site, and the defect
it found was one the regime site does not have.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfgarchon.alg.numerical.coupling.block_iterators import BlockIterator
from mfgarchon.alg.numerical.coupling.fictitious_play import FictitiousPlayIterator
from mfgarchon.alg.numerical.coupling.multi_population_iterator import MultiPopulationIterator
from mfgarchon.alg.numerical.coupling.newton_mfg_solver import NewtonMFGSolver
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.core.multi_population import MultiPopulationProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc

_NX, _NT = 12, 5


def _make_problem(sigma=0.3):
    components = MFGComponents(
        hamiltonian=SeparableHamiltonian(
            control_cost=QuadraticControlCost(control_cost=1.0),
            coupling=lambda m: m,
            coupling_dm=lambda m: 1.0,
        ),
        # NON-ZERO and non-constant, deliberately. With `u_terminal = 0` and a stub returning
        # zeros, `assert values[-1] == zeros` cannot separate "the terminal row was restored" from
        # "nothing ever wrote it", so the owner's terminal-restore invariant was unpinned at every
        # call site. Measured: with the old fixture, replacing `U_terminal` with `None` at all four
        # wired sites left the whole suite green.
        u_terminal=lambda x: 1.0 + np.asarray(x, dtype=float) * 0.0 + 3.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(
        geometry=TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[_NX], boundary_conditions=no_flux_bc(dimension=1)),
        T=0.2,
        Nt=_NT,
        sigma=sigma,
        components=components,
    )


class _DivergingHJB(HJBFDMSolver):
    """Returns a value function carrying a NaN, which is what a diverged inner Newton produces.

    One NaN, not a whole array of them: an all-NaN array would also be caught by a check that only
    looked at ``U[0, 0]``, and the point of the owner is that it looks at all of it.
    """

    def solve_hjb_system(self, *args, source_term=None, volatility_field=None, **kwargs):
        # Named because the base class refuses a bare **kwargs that would swallow them (#2020).
        #
        # `source_term` is accepted and ignored, and that is sound only because of what this stub
        # returns: a fixed diverged array, whatever it is asked. No source term can change a
        # constant. Asserting it None instead was measured to fail -- RegimeSwitchingIterator
        # passes its regime mass-transfer source here -- so the assertion would have been false
        # about the call sites rather than a guard on the stub.
        #
        # `volatility_field` IS asserted: no site under test supplies one, so a future site that
        # does should surface here rather than be silently dropped.
        assert volatility_field is None, "this stub cannot honour a volatility field"
        # 7.0, not 0.0: the fixture's terminal condition is 4.0, so a published U whose last row
        # is 7.0 means the restore never ran and one that is 4.0 means it did. A zero-filled stub
        # against a zero terminal is the non-discriminating pair this test used to have.
        U = np.full((_NT + 1, _NX), 7.0)
        U[_NT // 2, _NX // 2] = np.nan
        return U


def _run_to_completion(call):
    """Call it, and fail loudly if an FP-side exception escaped instead.

    This is the assertion that discriminates. Pre-#1718 the FP solver raises from inside the loop.
    """
    try:
        return call()
    except Exception as exc:
        pytest.fail(
            f"a diverged HJB escaped as {type(exc).__name__} from inside the coupling loop instead "
            f"of being attributed to HJB: {exc}"
        )


# --- the owner ----------------------------------------------------------------------------------
#
# The owner is imported HERE rather than at module scope, so that this file still COLLECTS against
# the pre-#1718 tree. Imported at the top it does not: the name does not exist there, the module
# errors during collection, and the call-site tests below never run -- turning a behavioural
# discrimination check into "the symbol is missing", which proves only that something was added.


def _owner():
    from mfgarchon.alg.numerical.coupling.fixed_point_utils import (
        diverged_value_function,
        value_function_is_finite,
    )

    return diverged_value_function, value_function_is_finite


def test_finite_value_function_returns_none():
    diverged_value_function, value_function_is_finite = _owner()
    assert diverged_value_function(np.ones((3, 4)), np.arange(4.0), site="probe") is None
    assert value_function_is_finite(np.ones((3, 4))) is True


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_every_non_finite_value_is_caught(bad):
    diverged_value_function, value_function_is_finite = _owner()
    U = np.ones((3, 4))
    U[1, 2] = bad
    assert value_function_is_finite(U) is False
    assert diverged_value_function(U, np.arange(4.0), site="probe") is not None


def test_the_diverged_iterate_is_published_with_its_terminal_row_restored():
    diverged_value_function, _ = _owner()
    U = np.ones((3, 4))
    U[0, 0] = np.nan
    terminal = np.arange(4.0)
    out = diverged_value_function(U, terminal, site="probe", iteration=2)
    assert np.isnan(out[0, 0]), "the NaN must survive: it is the diagnostic"
    np.testing.assert_array_equal(out[-1], terminal)
    assert np.isnan(U[0, 0]), "the caller's NaN must survive; the array is not ours to clean"
    assert not np.array_equal(U[-1], terminal), (
        "the caller's terminal row must not be written; the array is the HJB solver's own return "
        "value, not one this loop owns"
    )


def test_a_missing_terminal_condition_is_not_written_into_the_array():
    """NewtonMFGSolver's residual carries an optional terminal condition."""
    diverged_value_function, _ = _owner()
    U = np.ones((3, 4))
    U[0, 0] = np.nan
    out = diverged_value_function(U, None, site="probe")
    np.testing.assert_array_equal(out[-1], np.ones(4))


# --- the call sites -----------------------------------------------------------------------------


@pytest.mark.parametrize("IterCls", [BlockIterator, FictitiousPlayIterator])
def test_single_pair_iterators_stop_and_attribute(IterCls):
    problem = _make_problem()
    it = IterCls(problem, _DivergingHJB(problem), FPFDMSolver(problem))
    result = _run_to_completion(lambda: it.solve(max_iterations=3, tolerance=1e-8))
    U = result[0] if isinstance(result, tuple) else result.U
    assert not np.all(np.isfinite(U)), (
        "the diverged iterate must be published, not replaced by the last finite one: a caller "
        "reading output validation rather than the convergence reason would otherwise see a "
        "clean result"
    )
    (
        np.testing.assert_array_equal(np.asarray(U)[-1], np.full(_NX, 4.0)),
        (
            "the terminal row must be restored from the problem's terminal condition (4.0), not "
            "left as the diverged solver's own output (7.0)"
        ),
    )


def test_multi_population_stops_at_the_first_diverged_population():
    probs = [_make_problem(), _make_problem()]
    multi = MultiPopulationProblem(populations=probs, population_names=["P0", "P1"])
    it = MultiPopulationIterator(multi, [_DivergingHJB(p) for p in probs], [FPFDMSolver(p) for p in probs])
    result = _run_to_completion(lambda: it.solve(max_iterations=3, tolerance=1e-8))
    assert result.converged is False
    assert not np.all(np.isfinite(result.U[0])), "population 0's diverged iterate must be published"
    (
        np.testing.assert_array_equal(np.asarray(result.U[0])[-1], np.full(_NX, 4.0)),
        ("population 0's terminal row must be restored, not left at the stub's 7.0"),
    )


class _LateDivergingHJB(HJBFDMSolver):
    """Finite on the first sweep, NaN from the second.

    `_DivergingHJB` diverges immediately, which only exercises the sweep-0 path. The stale-error
    defect lives on the OTHER side of that branch: `errors` is re-bound by every COMPLETED sweep,
    so it is a divergence at sweep >= 1 that can publish the previous sweep's values.
    """

    def __init__(self, problem):
        super().__init__(problem)
        self._sweep = 0

    def solve_hjb_system(self, *args, source_term=None, volatility_field=None, **kwargs):
        assert volatility_field is None, "this stub cannot honour a volatility field"
        self._sweep += 1
        U = np.full((_NT + 1, _NX), 7.0)
        if self._sweep > 1:
            U[_NT // 2, _NX // 2] = np.nan
        return U


def test_a_late_divergence_does_not_publish_the_previous_sweeps_errors():
    """The #1672 shape: the best-looking number attached to the worst solve.

    `errors` is assigned in the convergence block at the END of a sweep, so a divergence at sweep
    >= 1 breaks out with the PREVIOUS completed sweep's values still bound. Measured before the
    clearing line existed: `converged=False, iterations=2, errors=[2.2204e-16, 2.2204e-16]` -- a
    converged-looking per-population error for a sweep in which nothing was measured and no FP ran.

    This test exists because that clearing was UNPINNED: removing the line left the whole
    CI-marker suite green (2741 passed), while the PR claimed each fix had been verified by
    reverting it. It had not been, for this one.
    """
    probs = [_make_problem(), _make_problem()]
    multi = MultiPopulationProblem(populations=probs, population_names=["P0", "P1"])
    it = MultiPopulationIterator(multi, [_LateDivergingHJB(p) for p in probs], [FPFDMSolver(p) for p in probs])
    result = _run_to_completion(lambda: it.solve(max_iterations=4, tolerance=1e-30))

    assert result.converged is False
    assert result.iterations >= 2, "the fixture must reach a sweep beyond the first"
    assert result.errors == [], (
        f"a diverged sweep published errors={result.errors} -- the previous completed sweep's "
        f"values, beside iterations={result.iterations} counting the sweep that measured nothing"
    )


def test_regime_switching_stops_and_attributes():
    from mfgarchon.alg.numerical.coupling.regime_switching_iterator import RegimeSwitchingIterator
    from mfgarchon.core.regime_switching import RegimeSwitchingConfig

    p0, p1 = _make_problem(), _make_problem()
    config = RegimeSwitchingConfig(transition_matrix=np.array([[-0.1, 0.1], [0.2, -0.2]]))
    it = RegimeSwitchingIterator(
        problems=[p0, p1],
        regime_config=config,
        hjb_solvers=[_DivergingHJB(p0), _DivergingHJB(p1)],
        fp_solvers=[FPFDMSolver(p0), FPFDMSolver(p1)],
    )
    result = _run_to_completion(lambda: it.solve())
    assert result.converged is False
    assert not np.all(np.isfinite(result.values[0]))
    (
        np.testing.assert_array_equal(np.asarray(result.values[0])[-1], np.full(_NX, 4.0)),
        ("regime 0's terminal row must be restored"),
    )
    # The regimes AFTER the diverged one were never solved. `Us_new` initialises to None, so a
    # result that published it would hand a consumer None where the contract promises an
    # (Nt+1, Nx) array -- and a consumer looping over `values` to FIND the NaN is exactly what
    # publishing the diverged iterate is for.
    assert result.values[1] is not None, "unsolved regimes must not be published as None"
    assert np.asarray(result.values[1]).ndim == 2, (
        f"unsolved regime published with shape {np.asarray(result.values[1]).shape}, against a documented (Nt+1, Nx)"
    )
    assert np.asarray(result.values[1])[-1].shape == (_NX,)
    for j, d in enumerate(result.densities):
        assert np.asarray(d).ndim == 2, f"density {j} published 1-D against a documented (Nt+1, Nx)"


def test_newton_warmup_stops_the_solve_before_fp_sees_the_diverged_u():
    """The site whose guard was inert, and the only one that needed a spy to see it.

    `_run_picard_warmup`'s `break` leaves only the warmup loop. `solve()` then calls
    `compute_residual_norm(U, M)`, which composes an FP solve from U -- so the diverged value
    function reached FP anyway and raised the same CFL misattribution, and the site's observable
    behaviour was byte-identical before and after the guard was added.

    Asserting `converged is False` alone would NOT catch that: the run raised, so the test would
    error rather than fail, and on a version that swallowed the exception it would pass. What pins
    it is counting FP invocations and checking their input.
    """
    problem = _make_problem()
    calls = []
    original = FPFDMSolver.solve_fp_system

    def spy(self, m_initial, *a, **kw):
        arrays = [x for x in list(a) + list(kw.values()) if isinstance(x, np.ndarray)]
        calls.append(any(not np.all(np.isfinite(x)) for x in arrays))
        return original(self, m_initial, *a, **kw)

    FPFDMSolver.solve_fp_system = spy
    try:
        solver = NewtonMFGSolver(problem, _DivergingHJB(problem), FPFDMSolver(problem), picard_warmup=2)
        U, _M, info = _run_to_completion(lambda: solver.solve(max_iterations=3, tolerance=1e-8, verbose=False))
    finally:
        FPFDMSolver.solve_fp_system = original

    assert not any(calls), (
        f"the FP solver was handed a non-finite value function ({sum(calls)} of {len(calls)} "
        f"calls). The warmup guard stopped the warmup loop but not the solve."
    )
    assert info["converged"] is False
    assert info["convergence_reason"] == "diverged_nan"
    assert not np.all(np.isfinite(U)), "the diverged iterate must still be published"
    # The assertion the other call-site tests carry and this one was missing -- at the very site
    # this commit exists to repair. `mfg_residual.U_terminal` is a real array here, so the
    # mutation U_terminal -> None is not vacuous: it changes what the solver publishes.
    (
        np.testing.assert_array_equal(np.asarray(U)[-1], np.full(_NX, 4.0)),
        (
            "the terminal row must be restored from the problem's terminal condition (4.0), not left "
            "as the diverged solver's own output (7.0)"
        ),
    )
    # The status block must describe THIS solve. `picard_warmup` is the configured count and
    # `newton_residuals` can hold a previous solve's history on the same instance.
    assert info["newton_iterations"] == 0, (
        f"reported {info['newton_iterations']} Newton iterations for a solve that ran none"
    )
    assert info["total_iterations"] <= info["picard_iterations"], (
        f"total_iterations={info['total_iterations']} below picard_iterations="
        f"{info['picard_iterations']} is self-contradicting"
    )


def test_graph_solver_stops_and_publishes_a_usable_result():
    """The site an earlier version of this file declared out of reach.

    That claim -- "needs a graph geometry this fixture cannot build" -- was false. There is no
    graph geometry: `tests/integration/test_graph_mfg_solver.py::_make_3node_system` builds the
    solver from three ordinary 1-D problems plus a 3x3 adjacency matrix, the same problem shape
    `_problem()` above already makes. The untested site was the one carrying the placeholder-array
    defect, and the false impossibility is what kept it from being measured.
    """
    from mfgarchon.alg.numerical.coupling.graph_coupling import AdjacencyCoupling
    from mfgarchon.alg.numerical.coupling.graph_mfg_solver import GraphMFGSolver

    probs = [_make_problem() for _ in range(3)]
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    it = GraphMFGSolver(
        problems=probs,
        coupling=AdjacencyCoupling(A, alpha=0.05, beta=0.02),
        hjb_solvers=[_DivergingHJB(p) for p in probs],
        fp_solvers=[FPFDMSolver(p) for p in probs],
    )
    result = _run_to_completion(lambda: it.solve())
    assert result.converged is False
    assert not np.all(np.isfinite(result.values[0])), "node 0's diverged iterate must be published"
    np.testing.assert_array_equal(np.asarray(result.values[0])[-1], np.full(_NX, 4.0))
    # Nodes 1 and 2 were never solved this sweep. `Us_new` initialises to np.empty(0), so
    # publishing it unchanged made `result.values[j][-1]` raise IndexError for every unsolved
    # node -- crashing the very consumer that loops over values to find the NaN.
    for j in (1, 2):
        arr = np.asarray(result.values[j])
        assert arr.ndim == 2, f"unsolved node {j} published with shape {arr.shape}, against a documented (Nt+1, Nx)"
        assert arr.shape[0] > 0, f"unsolved node {j} published a zero-length array"
        arr[-1]  # must not raise
    for j, d in enumerate(result.densities):
        assert np.asarray(d).ndim == 2, f"density {j} published 1-D against a documented (Nt+1, Nx)"
