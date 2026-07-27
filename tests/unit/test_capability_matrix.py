"""Pinning tests for the capability ratchet (scripts/capability_matrix.py).

The matrix records what the public solve surface can actually do, and its value comes
entirely from two properties that are easy to erode:

1. ``--check-baseline`` fails in BOTH directions. One-directional checking lets a
   recovery land unrecorded; the next baseline then encodes it as if it had always
   held, and the gate loses the ability to say when anything was fixed.
2. The ``--self-test`` mutation breaks the invariant the oracles measure. The first
   mutation written for it was a constant density scale, which left every drift cell
   PASS -- correctly, since ``max|mass(t) - mass(0)|`` is invariant under a uniform
   rescaling. A control that cannot fail proves nothing, which is the same defect the
   matrix exists to catch (Issues #1714, #1715).

These tests exercise the comparison and mutation logic directly. They do not solve
anything: the cells themselves are exercised by running the script.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "capability_matrix.py"


@pytest.fixture(scope="module")
def cm():
    spec = importlib.util.spec_from_file_location("capability_matrix", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline(**statuses):
    return {name: {"status": s, "artifact": {}} for name, s in statuses.items()}


# ---------------------------------------------------------------------------
# Baseline comparison -- the ratchet
# ---------------------------------------------------------------------------


def test_identical_statuses_report_no_problem(cm):
    base = _baseline(a="PASS", b="UNSUPPORTED")
    assert cm.compare_to_baseline({"a": "PASS", "b": "UNSUPPORTED"}, base) == []


def test_regression_is_caught(cm):
    problems = cm.compare_to_baseline({"a": "FAIL"}, _baseline(a="PASS"))
    assert len(problems) == 1
    assert "REGRESSION" in problems[0]


def test_recovery_is_also_caught(cm):
    """The direction that a one-way ratchet would silently allow."""
    problems = cm.compare_to_baseline({"a": "PASS"}, _baseline(a="UNSUPPORTED"))
    assert len(problems) == 1
    assert "RECOVERED" in problems[0]


def test_shift_between_two_non_pass_states_is_caught(cm):
    """UNSUPPORTED -> FAIL is neither regression nor recovery, and still must not pass.

    This is the transition that would hide a solver going from "refuses to run" to
    "runs and returns a wrong answer" -- strictly worse, and invisible to a check
    that only compares against PASS.
    """
    problems = cm.compare_to_baseline({"a": "FAIL"}, _baseline(a="UNSUPPORTED"))
    assert len(problems) == 1
    assert "SHIFT" in problems[0]


def test_deleted_cell_is_caught(cm):
    """Deleting a cell must not be a way to make a red baseline go green."""
    problems = cm.compare_to_baseline({}, _baseline(a="PASS"))
    assert len(problems) == 1
    assert "DISAPPEARED" in problems[0]


def test_new_cell_is_caught(cm):
    problems = cm.compare_to_baseline({"a": "PASS", "b": "PASS"}, _baseline(a="PASS"))
    assert len(problems) == 1
    assert "NEW cell" in problems[0]


def test_shipped_baseline_covers_every_declared_cell(cm):
    """A cell added without regenerating the baseline is a cell nothing watches."""
    with open(_SCRIPT.parent / "capability_baseline.json") as fh:
        baseline = json.load(fh)["cells"]
    assert set(baseline) == set(cm.CELLS), (
        "scripts/capability_baseline.json is out of sync with CELLS; regenerate with --write-baseline"
    )


def test_every_mass_oracle_cell_is_a_declared_cell(cm):
    assert set(cm.CELLS) >= cm.MASS_ORACLE_CELLS


def test_currently_failing_density_cells_are_still_registered_for_the_self_test(cm):
    """A cell that is not PASS today has an oracle the self-test cannot yet exercise.

    ``regime_switching/non_negativity`` raises before producing a density (#1681), so
    nothing has proved its oracle reads what it claims to. Listing it in
    MASS_ORACLE_CELLS is what makes the self-test pick it up the moment it recovers --
    without that, the recovery would land with an unproven oracle, which is exactly
    the state #1715 measured in the agreement tests.
    """
    density_cells = {
        "fdm_upwind/mass_conservation",
        "sl_linear/mass_conservation",
        "fdm_centered/mass_conservation",
        "fvm_muscl/mass_conservation",
        "fvm_vs_fdm/agreement",
        "regime_switching/non_negativity",
    }
    assert density_cells <= cm.MASS_ORACLE_CELLS, (
        f"unregistered density cells: {sorted(density_cells - cm.MASS_ORACLE_CELLS)}"
    )


# ---------------------------------------------------------------------------
# The self-test mutation -- the control has to be able to fail
# ---------------------------------------------------------------------------


def _drift(M):
    mass = M.sum(axis=1)
    return float(np.abs(mass - mass[0]).max())


def test_ramp_mutation_injects_drift(cm, monkeypatch):
    M = np.ones((5, 4))
    assert _drift(M) == 0.0
    monkeypatch.setattr(cm, "_DENSITY_MUTATION", 0.10)
    mutated = cm._apply_mutation(M)
    assert _drift(mutated) == pytest.approx(0.4, rel=1e-12)  # 10% of mass 4.0


def test_ramp_mutation_leaves_the_first_slice_alone(cm, monkeypatch):
    """t=0 is the reference the drift is measured against; moving it would mask drift."""
    M = np.ones((5, 4))
    monkeypatch.setattr(cm, "_DENSITY_MUTATION", 0.10)
    np.testing.assert_allclose(cm._apply_mutation(M)[0], M[0])


def test_the_mutation_does_what_a_constant_scale_cannot(cm, monkeypatch):
    """Pins the reason the mutation is a ramp.

    If someone replaces the ramp with ``M * factor``, the self-test goes back to
    reporting every drift cell INERT -- not because the cells broke, but because the
    control cannot break them.

    An earlier version of this test asserted only ``_drift(M * 1.5) == 0.0``, which
    never called the module under test: replacing the entire module with ``STUB = 1``
    left it green. It was a comment wearing a test's authority -- the exact bucket
    #1714 counts. It now compares the real mutation against the inert one.
    """
    M = np.ones((5, 4))
    assert _drift(M * 1.5) == 0.0, "a uniform scale cannot create drift -- that is the point"
    monkeypatch.setattr(cm, "_DENSITY_MUTATION", 0.10)
    assert _drift(cm._apply_mutation(M)) > 0.0


def test_mutation_is_off_by_default(cm):
    """A leaked mutation would make every cell FAIL and the baseline meaningless."""
    assert cm._DENSITY_MUTATION is None
    M = np.arange(20, dtype=float).reshape(5, 4)
    np.testing.assert_array_equal(cm._apply_mutation(M), M)


def test_mutation_applies_along_time_for_higher_dimensional_density(cm, monkeypatch):
    """2-D densities are (nt, nx, ny); the ramp must broadcast over time, not space."""
    M = np.ones((4, 3, 2))
    monkeypatch.setattr(cm, "_DENSITY_MUTATION", 0.30)
    mutated = cm._apply_mutation(M)
    assert mutated.shape == M.shape
    np.testing.assert_allclose(mutated[0], 1.0)
    np.testing.assert_allclose(mutated[-1], 1.30)


# ---------------------------------------------------------------------------
# Harness-error classification
# ---------------------------------------------------------------------------


def test_harness_breakage_is_error_not_unsupported(cm, monkeypatch):
    """A wrong signature must never read as "the library does not support this".

    This is the defect the gfdm_rbf cell shipped with on its first run: a missing
    positional argument surfaced as UNSUPPORTED, indistinguishable from the
    NotImplementedError the cell exists to record.
    """

    def broken():
        raise TypeError("f() missing 1 required positional argument: 'collocation_points'")

    monkeypatch.setattr(cm, "CELLS", {"broken/cell": broken})
    assert cm.evaluate()["broken/cell"]["status"] == "ERROR"


def test_a_refusal_by_the_code_under_test_is_unsupported(cm, monkeypatch):
    def refuses():
        raise NotImplementedError("derivative_method='rbf' is not supported (Issue #1553)")

    monkeypatch.setattr(cm, "CELLS", {"refuses/cell": refuses})
    result = cm.evaluate()["refuses/cell"]
    assert result["status"] == "UNSUPPORTED"
    assert result["artifact"]["exception"] == "NotImplementedError"
    assert "#1553" in result["artifact"]["message"]


# ---------------------------------------------------------------------------
# Phase separation -- classifying by exception type was the wrong shape
# ---------------------------------------------------------------------------


def test_a_valueerror_while_building_the_fixture_is_error_not_unsupported(cm, monkeypatch):
    """The library validates its own arguments with ValueError.

    So a mistyped fixture argument raises the same class as a genuine refusal, and
    under type-based classification read as "the library does not support this
    scheme". Only the phase separates them.
    """

    def bad_fixture():
        cm._construct("mistyped fixture", lambda: (_ for _ in ()).throw(ValueError("BC dimension mismatch")))

    monkeypatch.setattr(cm, "CELLS", {"bad/fixture": bad_fixture})
    result = cm.evaluate()["bad/fixture"]
    assert result["status"] == "ERROR"
    assert "constructing mistyped fixture" in result["artifact"]["message"]


def test_an_exception_while_computing_the_oracle_is_error_not_unsupported(cm, monkeypatch):
    """A solver returning a malformed result must not read as a library refusal.

    `M.shape[1]` on `np.asarray(None)` raises IndexError, which no denylist of
    "harness exception types" would plausibly have contained.
    """

    def bad_measure():
        class Result:
            M = None

        cm._measure("mass drift", cm._mass_drift, Result(), None)

    monkeypatch.setattr(cm, "CELLS", {"bad/measure": bad_measure})
    result = cm.evaluate()["bad/measure"]
    assert result["status"] == "ERROR"
    assert "measuring mass drift" in result["artifact"]["message"]


def test_a_refusal_from_the_solve_phase_survives_the_phase_split(cm, monkeypatch):
    """The split must not make everything ERROR -- UNSUPPORTED still has to be reachable."""

    def refuses():
        cm._construct("fine fixture", lambda: object())
        raise ValueError("FP solver: density went to -1.031e-03 at timestep 7/10")

    monkeypatch.setattr(cm, "CELLS", {"solve/refuses": refuses})
    assert cm.evaluate()["solve/refuses"]["status"] == "UNSUPPORTED"


# ---------------------------------------------------------------------------
# ERROR is never comparable and never baselined
# ---------------------------------------------------------------------------


def test_errored_lists_broken_cells(cm):
    results = {
        "a": {"status": "PASS", "artifact": {}},
        "b": {"status": "ERROR", "artifact": {}},
        "c": {"status": "UNSUPPORTED", "artifact": {}},
    }
    assert cm.errored(results) == ["b"]


def test_an_error_baselined_as_error_would_have_compared_equal(cm):
    """Why ERROR is gated before the baseline is read, not inside the comparison.

    `compare_to_baseline` is a status diff, so ERROR-vs-ERROR is a match. Had the
    gate lived only in the comparison, a harness broken during a `--write-baseline`
    regeneration would have been recorded as the expected state and stayed green
    forever -- the matrix no longer measuring anything, and reporting success for it.
    """
    assert cm.compare_to_baseline({"a": "ERROR"}, _baseline(a="ERROR")) == []


def _run_main(cm, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["capability_matrix.py", *argv])
    with pytest.raises(SystemExit) as exc:
        cm.main()
    return exc.value.code


def test_check_baseline_exits_2_on_error_without_reading_the_file(cm, monkeypatch, tmp_path):
    """The baseline path below does not exist -- reaching it would raise, not exit 2."""

    def boom():
        raise TypeError("signature drift")

    monkeypatch.setattr(cm, "CELLS", {"x/y": boom})
    assert _run_main(cm, monkeypatch, ["--check-baseline", str(tmp_path / "absent.json")]) == 2


def test_write_baseline_refuses_to_record_an_error(cm, monkeypatch, tmp_path):
    def boom():
        raise TypeError("signature drift")

    monkeypatch.setattr(cm, "CELLS", {"x/y": boom})
    target = tmp_path / "baseline.json"
    assert _run_main(cm, monkeypatch, ["--write-baseline", str(target)]) == 2
    assert not target.exists(), "a broken harness must not be able to bake itself into a baseline"


def test_a_healthy_tree_still_writes_and_matches(cm, monkeypatch, tmp_path):
    """The ERROR gate must not block the normal path."""
    monkeypatch.setattr(cm, "CELLS", {"x/y": lambda: ("PASS", {"n": 1})})
    target = tmp_path / "baseline.json"
    assert _run_main(cm, monkeypatch, ["--write-baseline", str(target)]) == 0
    assert _run_main(cm, monkeypatch, ["--check-baseline", str(target)]) == 0


def test_written_baselines_are_strict_json(cm, monkeypatch, tmp_path):
    """`allow_nan=False`. Python emits a bare `NaN` token otherwise, which is not JSON."""
    art = {"worst": float("nan"), "tolerance": 0.07, "all_finite": True}
    monkeypatch.setattr(cm, "CELLS", {"x/y": lambda: ("FAIL", art)})
    target = tmp_path / "b.json"
    assert _run_main(cm, monkeypatch, ["--write-baseline", str(target)]) == 1
    assert not target.exists()


def test_report_survives_a_partial_artifact(cm):
    """A reporting crash would lose the very run that shows something went wrong."""
    assert "NON-FINITE" in cm._summarise({"all_finite": False, "worst": None})
    assert cm._summarise({}) == "{}"
    assert "0.500%" in cm._summarise({"worst": 0.005})  # tolerance absent
    assert "TypeError" in cm._summarise({"exception": "TypeError"})  # message absent


# ---------------------------------------------------------------------------
# self_test driver
# ---------------------------------------------------------------------------


def test_self_test_aborts_when_the_harness_is_broken(cm, monkeypatch):
    def boom():
        raise TypeError("signature drift")

    monkeypatch.setattr(cm, "CELLS", {"x/y": boom})
    monkeypatch.setattr(cm, "MASS_ORACLE_CELLS", {"x/y"})
    assert cm.self_test() == 2


def test_self_test_is_inconclusive_rather_than_green_with_nothing_to_mutate(cm, monkeypatch):
    """Zero PASS cells means the control proved nothing. It must not exit 0."""
    monkeypatch.setattr(cm, "CELLS", {"x/y": lambda: ("FAIL", {})})
    monkeypatch.setattr(cm, "MASS_ORACLE_CELLS", {"x/y"})
    assert cm.self_test() == 1


def test_self_test_reports_an_inert_oracle(cm, monkeypatch):
    """A cell that ignores the mutation must fail the self-test, not pass it."""
    monkeypatch.setattr(cm, "CELLS", {"inert/cell": lambda: ("PASS", {})})
    monkeypatch.setattr(cm, "MASS_ORACLE_CELLS", {"inert/cell"})
    assert cm.self_test() == 1


def test_self_test_passes_when_the_oracle_reads_the_mutation(cm, monkeypatch):
    def honest():
        M = cm._apply_mutation(np.ones((5, 4)))
        return ("PASS" if _drift(M) == 0.0 else "FAIL"), {}

    monkeypatch.setattr(cm, "CELLS", {"honest/cell": honest})
    monkeypatch.setattr(cm, "MASS_ORACLE_CELLS", {"honest/cell"})
    assert cm.self_test() == 0


def test_self_test_restores_the_mutation_even_if_a_cell_raises(cm, monkeypatch):
    """A leaked mutation would make every later run FAIL for no reason."""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            return "PASS", {}
        raise RuntimeError("solver blew up under mutation")

    monkeypatch.setattr(cm, "CELLS", {"flaky/cell": flaky})
    monkeypatch.setattr(cm, "MASS_ORACLE_CELLS", {"flaky/cell"})
    cm.self_test()
    assert cm._DENSITY_MUTATION is None
