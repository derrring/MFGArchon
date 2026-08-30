"""Performance monitoring reports the solve's actual verdict (#1684 item 4).

Both convergence read sites in `utils/performance/monitoring.py` used to read
`convergence_achieved` off the SOLVER object. No class in this package has ever assigned that
attribute -- measured, four read sites and zero writes, while its sibling `iterations_run` is
real and set in `block_iterators.py` and `fictitious_play.py`. So it was never a rename; it is
the unimplemented half of a pair.

The consequences differed by site and neither surfaced as an error:

- the decorator read it through `getattr(solver, ..., False)`, so every tracked run was recorded
  as `converged: False` beside a genuine iteration count, whatever the solve did;
- `benchmark_solver` read it through `hasattr`, which was always false, so `run_data["converged"]`
  was never written at all -- and the loop discarded `solver.solve()`'s return value, which is
  where the verdict actually lives.

WHAT THESE TESTS CATCH, measured by restoring the pre-fix `monitoring.py` and re-running:
**5 of 6 fail.** The survivor is `test_the_decorator_reports_the_result_verdict[False]`, and it
survives for the reason that makes this defect worth a test at all -- the old code reported False
unconditionally, so on a non-converged solve it was accidentally right. A suite asserting only
that a converged run reports True would have caught this; one asserting only the False case would
not have. `test_the_reported_verdict_moves_with_the_solve` is immune to that asymmetry because it
compares the two runs against each other, which is the counterfactual #1684 uses throughout:
change the solve's outcome and show the reported flag MOVES.

The third test pins the distinction the fix introduced: a result carrying NO verdict omits the key
rather than defaulting it to False. Absent means not measured, False means measured and negative,
and the old code collapsed those two into the second.
"""

import pytest

import numpy as np

from mfgarchon.utils.performance.monitoring import PerformanceMonitor, benchmark_solver
from mfgarchon.utils.solver_result import SolverResult


def _result(converged: bool) -> SolverResult:
    return SolverResult(
        U=np.zeros((2, 3)),
        M=np.zeros((2, 3)),
        iterations=1,
        error_history_U=np.array([1.0]),
        error_history_M=np.array([1.0]),
        solver_name="stub",
        converged=converged,
    )


class _Solver:
    """Minimal stand-in carrying `iterations_run`, which is what selects the branch under test."""

    def __init__(self, problem=None, converged: bool = True):
        self.problem = problem
        self.iterations_run = 7
        self._converged = converged

    def solve(self):
        return _result(self._converged)


def _tracked_convergence_info(tmp_path, converged: bool) -> dict:
    monitor = PerformanceMonitor(storage_path=tmp_path)
    solver = _Solver(converged=converged)

    @monitor.performance_tracked(method_name="probe", update_baseline=False)
    def run(s):
        return s.solve()

    run(solver)
    recorded = monitor.metrics_history["probe"]
    assert len(recorded) == 1, "the decorator should record exactly one run"
    return recorded[0].convergence_info


@pytest.mark.parametrize("converged", [True, False])
def test_the_decorator_reports_the_result_verdict(tmp_path, converged):
    info = _tracked_convergence_info(tmp_path, converged)
    assert info["iterations"] == 7, "the real sibling attribute must still be read from the solver"
    assert info["converged"] is converged


def test_the_reported_verdict_moves_with_the_solve(tmp_path):
    """The counterfactual. Against the old code both runs reported False."""
    yes = _tracked_convergence_info(tmp_path / "a", True)["converged"]
    no = _tracked_convergence_info(tmp_path / "b", False)["converged"]
    assert (yes, no) == (True, False), (
        f"the reported verdict did not follow the solve: converged run reported {yes!r}, "
        f"non-converged run reported {no!r}"
    )


def test_a_result_without_a_verdict_omits_the_key(tmp_path):
    """Absent is not False. The old code could only produce False."""

    class _Bare:
        pass

    class _SolverBare(_Solver):
        def solve(self):
            return _Bare()

    monitor = PerformanceMonitor(storage_path=tmp_path)

    @monitor.performance_tracked(method_name="bare", update_baseline=False)
    def run(s):
        return s.solve()

    run(_SolverBare())
    info = monitor.metrics_history["bare"][0].convergence_info
    assert info["iterations"] == 7
    assert "converged" not in info, f"a verdict was invented where none was measured: {info!r}"


@pytest.mark.parametrize("converged", [True, False])
def test_benchmark_solver_records_the_verdict(converged):
    """The second site. Against the old code the key was absent for BOTH values."""

    def factory(problem, **config):
        return _Solver(problem, converged=converged)

    factory.__name__ = "StubSolver"
    out = benchmark_solver(factory, problem=object(), config_variations=[{}], repetitions=1)
    run_data = out["configurations"][0]["runs"][0]
    assert run_data["iterations"] == 7
    assert "converged" in run_data, "benchmark_solver recorded no verdict at all"
    assert run_data["converged"] is converged
