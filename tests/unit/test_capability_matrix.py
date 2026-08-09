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

import ast
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


def test_every_scheme_with_a_1d_cell_has_a_2d_cell(cm):
    """The gap #1745 exposed: every cell was 1-D, so a scheme could conserve mass to
    2.2e-16 in one dimension and not run at all in two with the matrix silent.

    Pinned as a rule rather than a list: a new scheme cell arriving in 1-D only
    reopens exactly that hole.
    """
    one_d = {c.split("/")[0] for c in cm.CELLS if "/mass_conservation" in c and not c.endswith("_2d/mass_conservation")}
    two_d = {c.split("/")[0].removesuffix("_2d") for c in cm.CELLS if c.endswith("_2d/mass_conservation")}
    assert one_d <= two_d, f"schemes with a 1-D mass cell and no 2-D sibling: {sorted(one_d - two_d)}"


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
        "sl_linear_2d/mass_conservation",
        "fdm_upwind_2d/mass_conservation",
        "fdm_centered_2d/mass_conservation",
        "fvm_muscl_2d/mass_conservation",
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

        cm._measure("mass drift", lambda: cm._mass_drift(Result(), None))

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


# ---------------------------------------------------------------------------
# The two structural fixes must not be able to revert silently
# ---------------------------------------------------------------------------


def test_measure_rejects_an_oracle_that_returns_only_the_artifact(cm):
    """The shape check is what keeps the verdict inside the wrapper.

    A cell whose oracle returns a bare artifact has its verdict computed one line
    outside ``_measure`` again, where a KeyError reads as a library refusal. Tests
    cannot catch that by construction -- they pass their own thunks -- so the
    wrapper refuses the old shape instead.
    """
    with pytest.raises(cm.HarnessError, match="must return \\(status, artifact\\)"):
        cm._measure("mass drift", lambda: {"mass_t0": 1.0})


def test_measure_rejects_an_unknown_status(cm):
    with pytest.raises(cm.HarnessError, match="must return"):
        cm._measure("x", lambda: ("PROBABLY_FINE", {}))


def test_measure_accepts_the_correct_shape(cm):
    """The guard must not reject what the production cells actually return."""
    assert cm._measure("x", lambda: ("PASS", {"n": 1})) == ("PASS", {"n": 1})


def test_a_keyerror_anywhere_in_a_cell_is_error_not_unsupported(cm, monkeypatch):
    """Second net, for the shape check being bypassed some other way.

    A partial artifact is never a library refusal, whichever phase surfaced it.
    """

    def raises_keyerror():
        raise KeyError("mass_t0")

    monkeypatch.setattr(cm, "CELLS", {"k/e": raises_keyerror})
    assert cm.evaluate()["k/e"]["status"] == "ERROR"


def test_the_harness_diagnostic_goes_to_stderr(cm, monkeypatch, tmp_path, capsys):
    """stdout carries the --json blob; a human block appended to it breaks parsing."""

    def boom():
        raise TypeError("signature drift")

    monkeypatch.setattr(cm, "CELLS", {"x/y": boom})
    monkeypatch.setattr(sys, "argv", ["capability_matrix.py", "--json"])
    with pytest.raises(SystemExit):
        cm.main()
    captured = capsys.readouterr()
    json.loads(captured.out)  # raises if the diagnostic leaked into stdout
    assert "Harness is broken" in captured.err


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


def test_self_test_restores_the_mutation(cm, monkeypatch):
    """A leaked mutation would make every later run FAIL for no reason.

    Named for what it checks. The earlier name said "even if a cell raises", which
    it did not earn: ``evaluate`` swallows every per-cell exception, so the
    ``finally`` is never reached via an exception and deleting the ``try/finally``
    killed no test. The reset itself is real and is what is pinned here.
    """
    monkeypatch.setattr(cm, "CELLS", {"cell": lambda: ("PASS", {})})
    monkeypatch.setattr(cm, "MASS_ORACLE_CELLS", {"cell"})
    cm.self_test()
    assert cm._DENSITY_MUTATION is None


def test_self_test_aborts_when_the_harness_breaks_under_mutation(cm, monkeypatch):
    """ERROR under mutation is not evidence the oracle bites.

    Scoring on "not PASS" counted a harness that broke under mutation as a working
    control -- the same shape as treating ERROR as a comparable status, surviving in
    the one place the earlier fix did not reach. It matters more here because this
    IS the control: it is what certifies the other cells still discriminate.
    """
    calls = {"n": 0}

    def healthy_then_broken():
        calls["n"] += 1
        if calls["n"] == 1:
            return "PASS", {}
        raise TypeError("apparatus broke under mutation")

    monkeypatch.setattr(cm, "CELLS", {"cell": healthy_then_broken})
    monkeypatch.setattr(cm, "MASS_ORACLE_CELLS", {"cell"})
    assert cm.self_test() == 2, "a broken apparatus must not certify the control"


def test_summarise_never_raises_on_any_shape(cm):
    """`never raises` has to hold for shapes nobody anticipated -- that is the point."""
    shapes = [
        {},
        {"min_density": None},
        {"max_drift": "not a float", "min_density": 0.0},
        {"worst": "text"},
        {"exception": 123},
        {"all_finite": 0},
        {"nested": {"a": [1, 2]}},
        {"obj": object()},
        {"worst": 0.01, "tolerance": "seven percent"},
    ]
    for art in shapes:
        assert isinstance(cm._summarise(art), str), f"raised or returned non-str for {art!r}"


def test_a_note_is_dropped_when_the_failure_itself_changes(cm, monkeypatch, tmp_path, capsys):
    """Carry-forward must not preserve an annotation across a different failure.

    The fixture below is the ORIGINAL defect, not a synthetic stand-in: the #1745 note cited
    residual 2.42e-01 beside an artifact that had since become 1.17e-05. The first version of
    this gate compared only `artifact["exception"]`, and both sides of that case are
    ConvergenceError -- so it carried the stale note forward and the run still reported zero
    unexplained cells. A guard checked against a case invented from the description of a defect
    will miss the defect; it has to be checked against the instance.
    """
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "cells": {
                    "fdm_centered_2d/mass_conservation": {
                        "status": "UNSUPPORTED",
                        "artifact": {
                            "exception": "ConvergenceError",
                            "message": "newton solver failed to converge after 30 iterations (residual: 2.42e-01)",
                        },
                        "intended": "DEFECT - identical residual 2.42e-01, so it is DIVERGING.",
                    }
                }
            }
        )
    )

    class ConvergenceError(RuntimeError):
        """Same TYPE as the baseline artifact records. That is the whole point of the case:
        exception-type comparison cannot tell these two failures apart."""

    def same_exception_different_residual():
        raise ConvergenceError("newton solver failed to converge after 30 iterations (residual: 1.17e-05)")

    monkeypatch.setattr(cm, "CELLS", {"fdm_centered_2d/mass_conservation": same_exception_different_residual})
    _run_main(cm, monkeypatch, ["--write-baseline", str(baseline)])

    rewritten = json.loads(baseline.read_text())["cells"]["fdm_centered_2d/mass_conservation"]
    assert "1.17e-05" in rewritten["artifact"]["message"], "the artifact must be refreshed"
    assert "intended" not in rewritten, (
        "the note said 2.42e-01 and DIVERGING; the cell now records 1.17e-05. Same exception "
        "type, different failure -- the note must be dropped so the cell reads as unexplained"
    )


def test_a_note_survives_a_regeneration_that_changes_nothing(cm, monkeypatch, tmp_path):
    """The counterpart: gating on the whole artifact must not throw away still-valid notes.

    Without this, the gate above would be indistinguishable from deleting carry-forward.
    """
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "cells": {
                    "x/y": {
                        "status": "UNSUPPORTED",
                        "artifact": {"exception": "ValueError", "message": "mass gate refused"},
                        "intended": "INTENDED - the guard is doing its job",
                    }
                }
            }
        )
    )

    def same_failure():
        raise ValueError("mass gate refused")

    monkeypatch.setattr(cm, "CELLS", {"x/y": same_failure})
    _run_main(cm, monkeypatch, ["--write-baseline", str(baseline)])

    rewritten = json.loads(baseline.read_text())["cells"]["x/y"]
    assert rewritten["intended"] == "INTENDED - the guard is doing its job"


def test_json_mode_emits_json_and_nothing_else(cm, monkeypatch, capsys):
    """`--json` must be parseable by a non-Python reader.

    The cells run real coupled solves, which write to stdout from three independent places --
    library INFO logging, Rich progress bars, and plain prints -- so the flag never produced
    parseable output. Suppressing them one at a time leaves the next addition free to break it
    again, so the whole evaluation is redirected; this pins the property, not the mechanism.
    """

    def noisy_pass():
        print("INFO: solver chatter that would corrupt the stream")
        return "PASS", {"note": "fine"}

    monkeypatch.setattr(cm, "CELLS", {"x/y": noisy_pass})
    _run_main(cm, monkeypatch, ["--json"])

    parsed = json.loads(capsys.readouterr().out)
    assert set(parsed) == {"x/y"}
    assert parsed["x/y"]["status"] == "PASS"


def test_the_baseline_comment_names_no_flag_that_does_not_exist(cm, monkeypatch, tmp_path):
    """The written `_comment` told readers to "See --explain."; argparse has no such flag.

    A generated artifact that instructs the reader to run a command which exits 2 is the same
    defect class as an error message naming a fix that does not work. Asks the parser the script
    actually builds, so the check cannot drift from what the CLI accepts.
    """
    import argparse

    seen = {}
    real_parse = argparse.ArgumentParser.parse_args

    def capture(self, *a, **kw):
        seen["flags"] = set(self._option_string_actions)
        return real_parse(self, *a, **kw)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)

    baseline = tmp_path / "fresh.json"
    monkeypatch.setattr(cm, "CELLS", {"x/y": lambda: ("PASS", {})})
    _run_main(cm, monkeypatch, ["--write-baseline", str(baseline)])

    declared = seen["flags"]
    assert "--write-baseline" in declared, "the capture did not observe the real parser"

    comment = json.loads(baseline.read_text())["_comment"]
    # `--` alone is the prose dash this comment uses, not a flag.
    cited = {tok.rstrip(".,;:") for tok in comment.split() if tok.startswith("--") and len(tok) > 2}
    assert cited, "nothing cited would make this check pass vacuously"
    assert cited <= declared, f"_comment cites flags the CLI does not accept: {sorted(cited - declared)}"


def test_json_stays_parseable_when_a_cell_has_no_note(cm, monkeypatch, tmp_path, capsys):
    """`--json` must emit JSON in the state the unexplained-cell report exists to describe.

    Routing prints one at a time is how this broke twice: a fix said "both remaining prints" when
    there were four, and the two it missed fire only when a non-PASS cell lacks a note -- exactly
    when the new feature has something to say. The state below is reached through the documented
    CLI, not by hand-editing: a baseline written to a NEW path carries no notes forward.
    """
    fresh = tmp_path / "fresh.json"
    monkeypatch.setattr(cm, "CELLS", {"x/y": lambda: (_ for _ in ()).throw(ValueError("refused"))})
    _run_main(cm, monkeypatch, ["--write-baseline", str(fresh)])
    capsys.readouterr()

    monkeypatch.setattr(cm, "CELLS", {"x/y": lambda: (_ for _ in ()).throw(ValueError("refused"))})
    _run_main(cm, monkeypatch, ["--json", "--check-baseline", str(fresh)])

    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert set(parsed) == {"x/y"}


def test_the_unexplained_count_is_derived_not_asserted(cm, monkeypatch, tmp_path, capsys):
    """The headline metric this feature introduces had no test; it could be wired to a constant.

    Two cells, one annotated and one not, so a hardcoded 0 or a hardcoded len() both fail.
    """
    baseline = tmp_path / "b.json"
    baseline.write_text(
        json.dumps(
            {
                "cells": {
                    "a/annotated": {
                        "status": "UNSUPPORTED",
                        "artifact": {"exception": "ValueError", "message": "refused"},
                        "intended": "INTENDED - the guard fired",
                    },
                    "b/bare": {
                        "status": "UNSUPPORTED",
                        "artifact": {"exception": "ValueError", "message": "refused"},
                    },
                }
            }
        )
    )

    def refused():
        raise ValueError("refused")

    monkeypatch.setattr(cm, "CELLS", {"a/annotated": refused, "b/bare": refused})
    _run_main(cm, monkeypatch, ["--check-baseline", str(baseline)])

    report = capsys.readouterr().out
    assert "1 non-PASS cell(s) with no recorded reason" in report, report
    assert "b/bare" in report
    assert "a/annotated" not in report.split("no recorded reason")[-1]


def test_every_cell_that_solves_records_the_picard_verdict():
    """A cell that solves must say whether the solve converged (#1871).

    The field exists because the mass oracles cannot see convergence -- mass is conserved on
    whatever drift field the FP step is handed. It is recorded, not gated: nothing in
    `--check-baseline` reacts to it, and a `_picard_verdict` returning constants leaves the
    full gate green (verified by mutation). This test is the only oracle over the field.

    It reads the SOURCE, not the baseline. The first version of this test compared recorded
    artifacts, and deleting a `_picard_verdict` call left it green -- the baseline still
    carried the field from the last regeneration, so the test pinned a record rather than
    the code that writes it. That is the same defect class the field itself exists to
    expose, one level up.

    The rule is structural, so there is no exemption list to keep in sync: a cell function
    that binds the result of a `.solve(...)` call must mention `_picard_verdict`. Cells that
    only construct a solver -- `_gfdm_rbf_cell` -- bind no result and are exempt by shape,
    which matters because it is why "no exception implies a verdict" cannot be the rule.
    """
    tree = ast.parse(_SCRIPT.read_text())

    def solves(node: ast.FunctionDef) -> bool:
        return any(
            isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "solve"
            for n in ast.walk(node)
        )

    solving = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and solves(n)]
    assert solving, "no cell binds a solve result -- the rule below would be vacuous"

    missing = sorted(
        n.name for n in solving if "_picard_verdict" not in ast.dump(n) and "_picard_verdict" not in ast.unparse(n)
    )
    assert not missing, f"these solve a problem and never record whether it converged: {missing}"
