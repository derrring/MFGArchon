"""Pinning tests for the discrimination ratchet (scripts/test_discrimination.py).

The sweep itself takes ~26 minutes (seven full-suite runs), so it is a weekly job, not
a gate. These tests pin the parts that decide what the sweep MEANS, which is where it
can silently stop working:

- the three-way verdict, which separates "no test covers this convention" (a finding)
  from "the mutation never ran" (a harness fault). An earlier design had only two
  outcomes and reported four mutations killing 113, 17, 5 and 5 tests as INEFFECTIVE
  because their control test-path had been guessed wrong -- while the one genuinely
  uncovered convention looked identical to them.
- the ratchet, which must fail in both directions for the same reason the capability
  baseline does.
- the mutation anchors, which are literal source text and rot when the source moves.
"""

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "test_discrimination.py"
_BASELINE = _SCRIPT.parent / "discrimination_baseline.json"


@pytest.fixture(scope="module")
def td():
    spec = importlib.util.spec_from_file_location("test_discrimination", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_discrimination"] = module
    spec.loader.exec_module(module)
    return module


def _base(**counts):
    return {"mutations": {n: {"owner": "x", "status": "ok", "kill_count": c} for n, c in counts.items()}}


def _now(**counts):
    return {n: {"owner": "x", "status": "ok", "kill_count": c, "killed": []} for n, c in counts.items()}


# ---------------------------------------------------------------------------
# The ratchet
# ---------------------------------------------------------------------------


def test_identical_counts_report_no_problem(td):
    assert td.compare_to_baseline(_now(a=10), _base(a=10)) == []


def test_a_dropped_kill_count_is_caught(td):
    """Discrimination lost: a convention 10 tests noticed is now noticed by 9."""
    problems = td.compare_to_baseline(_now(a=9), _base(a=10))
    assert len(problems) == 1
    assert "DISCRIMINATION LOST" in problems[0]


def test_a_raised_kill_count_is_also_caught(td):
    """The direction a one-way ratchet would swallow, taking the record with it."""
    problems = td.compare_to_baseline(_now(a=11), _base(a=10))
    assert len(problems) == 1
    assert "IMPROVED" in problems[0]


def test_a_mutation_going_ineffective_is_caught_before_its_count(td):
    """An unapplied mutation reports zero kills, which would read as total loss.

    It is a harness fault, not a coverage collapse, and must be named as one --
    otherwise the fix attempted would be to the tests rather than to the mutation.
    """
    now = _now(a=0)
    now["a"]["status"] = "INEFFECTIVE"
    problems = td.compare_to_baseline(now, _base(a=10))
    assert len(problems) == 1
    assert "INEFFECTIVE" in problems[0]
    assert "DISCRIMINATION LOST" not in problems[0]


def test_a_deleted_mutation_is_caught(td):
    """Deleting a mutation must not be a way to make a red ratchet go green."""
    problems = td.compare_to_baseline({}, _base(a=10))
    assert len(problems) == 1
    assert "DISAPPEARED" in problems[0]


def test_a_new_mutation_is_caught(td):
    problems = td.compare_to_baseline(_now(a=10, b=3), _base(a=10))
    assert len(problems) == 1
    assert "NEW mutation" in problems[0]


def test_an_uncovered_convention_holding_at_zero_is_not_a_regression(td):
    """UNCOVERED is a finding, but a stable one must not fail every run."""
    now = _now(a=0)
    now["a"]["status"] = "UNCOVERED"
    base = _base(a=0)
    base["mutations"]["a"]["status"] = "UNCOVERED"
    assert td.compare_to_baseline(now, base) == []


# ---------------------------------------------------------------------------
# The baseline that ships
# ---------------------------------------------------------------------------


def test_shipped_baseline_covers_every_declared_mutation(td):
    baseline = json.loads(_BASELINE.read_text())["mutations"]
    assert set(baseline) == {m.name for m in td.MUTATIONS}, (
        "scripts/discrimination_baseline.json is out of sync with MUTATIONS; regenerate with --write-baseline"
    )


def test_shipped_baseline_records_the_scope_it_was_measured_at(td):
    """A count whose scope is unrecorded is not comparable to anything.

    A baseline written under `--paths tests/unit` would otherwise be indistinguishable
    from one written over the whole tree, and a later run would read the difference as
    discrimination lost.
    """
    at = json.loads(_BASELINE.read_text()).get("_measured_at")
    assert isinstance(at, dict), "provenance must be structured, not a bare sha"
    for key in ("commit", "paths", "markers", "collected", "excluded"):
        assert at.get(key), f"_measured_at is missing {key!r}"
    assert at["markers"] == td.MARKERS, "baseline marker set has drifted from the script's"
    assert at["excluded"] == td.SELF_TESTS


def test_the_sweep_passes_its_self_exclusion_to_pytest(td, monkeypatch):
    """M1: the constant alone was pinned, the WIRING was not.

    Deleting the `--ignore` argument left all 19 pins green while the +1 contamination
    returned -- including the +1 that turns the one UNCOVERED convention into an
    apparently-covered one. This asserts the argument actually reaches pytest.
    """
    seen = {}

    class _Proc:
        stdout = "1 passed"
        returncode = 0

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(td.subprocess, "run", fake_run)
    td._pytest(["tests"])
    assert f"--ignore={td.SELF_TESTS}" in seen["cmd"], (
        f"the sweep did not exclude its own tests; argv was {seen['cmd']}"
    )


_RED_BASELINE = """\
............F...........                                                 [100%]
=================================== FAILURES ===================================
____________ TestBackendCreationEdgeCases.test_auto_selection_logging ____________
>       assert len(captured.out) > 0
E       assert 0 > 0
=========================== short test summary info ============================
FAILED tests/unit/test_backends/test_backend_factory.py::TestBackendCreationEdgeCases::test_auto_selection_logging
1 failed, 23 passed
"""


def test_the_refusal_reports_why_the_baseline_was_red(td, monkeypatch):
    """Wiring, not the helper: `_failure_excerpt` can be correct while nothing calls it.

    Same shape as M1 above. A refusal naming the node id but not the assertion costs a
    full CI round-trip, and buys nothing when the red does not reproduce locally.
    """

    class _Proc:
        stdout = _RED_BASELINE
        returncode = 1

    # The two startup guards shell out to git, and this test replaces subprocess
    # wholesale; they have their own pins above and are not what is under test here.
    monkeypatch.setattr(td, "_assert_clean_tree", lambda: None)
    monkeypatch.setattr(td, "_assert_import_is_the_mutated_tree", lambda: None)
    monkeypatch.setattr(td.subprocess, "run", lambda cmd, **kw: _Proc())
    monkeypatch.setattr(td.sys, "argv", ["test_discrimination.py"])
    with pytest.raises(SystemExit) as exc:
        td.main()
    message = str(exc.value)
    assert "assert 0 > 0" in message, f"the refusal named the test but not the assertion; it said:\n{message}"


_EMPTY_BASELINE = """\
============================= test session starts ==============================
collected 0 items

============================ no tests ran in 0.01s =============================
"""


def test_the_other_refusal_reports_why_too(td, monkeypatch):
    """Two exits call the excerpt; one test reaching one of them pins one of them.

    The `base.failed` branch fires first whenever stdout carries a FAILED line, so the test
    above can never reach this one. A `--paths` typo is the case that gets here -- pytest
    collects nothing, exits non-zero, and the FAILED set is empty -- and it is the case whose
    inline comment says an unnoticed one produces six bogus UNCOVERED findings.
    """

    class _Proc:
        stdout = _EMPTY_BASELINE
        returncode = 5

    monkeypatch.setattr(td, "_assert_clean_tree", lambda: None)
    monkeypatch.setattr(td, "_assert_import_is_the_mutated_tree", lambda: None)
    monkeypatch.setattr(td.subprocess, "run", lambda cmd, **kw: _Proc())
    monkeypatch.setattr(td.sys, "argv", ["test_discrimination.py"])
    with pytest.raises(SystemExit) as exc:
        td.main()
    message = str(exc.value)
    assert "collected 0 items" in message, (
        f"the second refusal reported the exit code but not the output behind it; it said:\n{message}"
    )


def test_the_excerpt_starts_at_the_failures_banner(td):
    """Everything before the banner is progress dots, which crowd out the reason."""
    assert td._failure_excerpt(_RED_BASELINE).startswith("=")
    assert "[100%]" not in td._failure_excerpt(_RED_BASELINE)


def test_a_long_excerpt_keeps_the_tail_and_says_what_it_dropped(td):
    """With -q the short summary lands last, so truncating the head is the safe end.

    Silent truncation would read as "that was the whole failure" -- the shape this
    repo files under a silent instrument.
    """
    long_output = "=" * 3 + " FAILURES " + "=" * 3 + "\n" + "\n".join(f"line {i}" for i in range(500))
    excerpt = td._failure_excerpt(long_output, limit=10)
    assert excerpt.splitlines()[-1] == "line 499", "the tail, where the summary is, was dropped"
    assert "omitted" in excerpt, "truncation was silent"


# ---------------------------------------------------------------------------
# The mutation anchors -- literal source text, so they rot when the source moves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", range(6))
def test_every_mutation_anchor_still_matches_exactly_once(td, index):
    """A silently unapplied mutation reports zero kills and reads as total blindness.

    `apply_mutation` raises on a miss, but only when the sweep is run -- which is
    weekly. This surfaces the rot on the next commit instead.
    """
    mut = td.MUTATIONS[index]
    source = (_SCRIPT.parents[1] / mut.path).read_text()
    assert source.count(mut.old) == 1, (
        f"mutation {mut.name!r}: anchor matches {source.count(mut.old)} times in {mut.path}, "
        f"expected exactly 1. The source moved; update the mutation."
    )


def test_the_mutation_list_matches_the_parametrisation(td):
    """The parametrize range above is a literal; this is what notices when it drifts."""
    assert len(td.MUTATIONS) == 6


def test_the_end_of_run_guard_checks_the_mutated_files_only(td, monkeypatch):
    """It must not fire on the script's own --json output.

    Implemented as a whole-tree check it did exactly that: `--json` wrote the kill
    matrix into the repo, the end-of-run assertion saw a modified tracked file, and
    `--write-baseline` in the same invocation was refused -- with a message blaming
    the operator for a file the script had just created. The startup check stays
    whole-tree; only this one is scoped.
    """
    seen = {}

    class _Proc:
        stdout = ""

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(td.subprocess, "run", fake_run)
    td._assert_mutations_restored(td.MUTATIONS)
    assert seen["cmd"][:4] == ["git", "status", "--porcelain", "--"], seen["cmd"]
    scoped = set(seen["cmd"][4:])
    assert scoped == {m.path for m in td.MUTATIONS}
    assert not any(p.endswith(".json") for p in scoped), "the guard must not watch its own outputs"


def test_main_uses_the_scoped_guard_at_the_end(td):
    """M1 again, third time: the function was pinned, the CALL SITE was not.

    Swapping the end-of-run call back to `_assert_clean_tree()` killed zero tests,
    because the test above exercises `_assert_mutations_restored` directly. Source
    inspection is the honest pin here -- driving `main()` means a 26-minute sweep --
    and it is the same technique the mutation-anchor tests already use.
    """
    body = inspect.getsource(td.main)
    assert "_assert_mutations_restored(selected)" in body, (
        "main() no longer uses the scoped end-of-run guard; a whole-tree check there "
        "fires on the script's own --json output and blocks --write-baseline"
    )
    # The whole-tree check belongs at startup and nowhere else: the run must begin from
    # a state it can prove it restored.
    assert body.count("_assert_clean_tree()") == 1


def test_the_kill_matrix_is_committed_beside_the_baseline(td):
    """The counts are the gate; the matrix is the evidence under them.

    Without it the population claims in #1715 -- "193 of 308 inert", "39 of 65" -- are
    unrecoverable by anyone, which is the shape of claim this whole campaign exists to
    stop shipping.
    """
    matrix = json.loads((_BASELINE.parent / "discrimination_killmatrix.json").read_text())
    # Owned by the module, not hand-added to the JSON: a hand-added key does not
    # survive the next --json, and this test caught exactly that regression once.
    assert matrix["_selection_regex_for_agreement_shaped"] == td.AGREEMENT_SHAPED
    assert matrix["_measured_at"] == json.loads(_BASELINE.read_text())["_measured_at"], (
        "matrix and baseline record different runs"
    )
    baseline = json.loads(_BASELINE.read_text())["mutations"]
    for name, res in baseline.items():
        assert matrix["mutations"][name]["kill_count"] == res["kill_count"], (
            f"{name}: matrix and baseline disagree -- they came from different runs"
        )


def test_the_sweep_excludes_this_file(td):
    """This file must not be inside the population it measures.

    `test_every_mutation_anchor_still_matches_exactly_once` fails under every mutation
    by design, so leaving it in adds +1 to every kill count. Measured before the
    exclusion: 129/34/19/5/5/0 became 130/35/20/6/6/1 -- and that last +1 turned the
    one UNCOVERED convention into an apparently-covered one, which is the instrument
    deleting its own finding.
    """
    assert td.SELF_TESTS == "tests/unit/test_discrimination_ratchet.py"
    assert str(Path(__file__).resolve()).endswith(td.SELF_TESTS)


def test_every_mutation_changes_something(td):
    for mut in td.MUTATIONS:
        assert mut.old != mut.new, f"{mut.name}: old and new are identical"


def test_every_mutation_declares_a_verify_expression(td):
    """Without it there is no way to tell a live mutation from an unapplied one."""
    for mut in td.MUTATIONS:
        assert mut.verify.strip(), f"{mut.name}: empty verify"


# --- killer-set ratchet (#1901): the half a kill COUNT cannot express -------------------

_MATRIX = {
    "mutations": {
        "drift_coefficient_2x": {
            "kill_count": 2,
            "killed": ["tests/a.py::test_alpha", "tests/a.py::test_beta"],
            "status": "ok",
        }
    }
}


def _results(killed, status="ok"):
    """Shaped as `main()` writes it -- key "killed", not "failed".

    The first version of this fabricated "failed", matching the reader under test rather
    than the producer. Both agreed, all six tests passed, and in production every killer
    read as departed because `main()` writes "killed". A fixture that carries its own copy
    of the data shape can only prove self-consistency; `test_the_reader_uses_the_key_main_writes`
    below is the structural guard, because no value-level test can catch this.
    """
    return {"drift_coefficient_2x": {"status": status, "kill_count": len(killed), "killed": sorted(killed)}}


def _baseline_for(count):
    return {"mutations": {"drift_coefficient_2x": {"kill_count": count, "status": "ok", "owner": "x"}}}


def test_an_equal_size_killer_swap_is_caught(td):
    """The motivating case. `drift_coefficient_2x` held 19 -> 19 across a one-for-one swap.

    One test stopped noticing the convention and a different one started. The count ratchet
    reported no change, which is why a count is a weaker instrument than it reads (#1901).
    """
    results = _results(["tests/a.py::test_alpha", "tests/a.py::test_gamma"])
    problems = td.compare_to_baseline(results, _baseline_for(2), _MATRIX)
    assert problems, "an equal-size swap passed: the killer-set ratchet is not wired in"
    assert any("STOPPED killing" in p and "test_beta" in p for p in problems), problems


def test_a_killer_leaving_is_a_loss_even_when_the_count_rises(td):
    """Two arriving and one leaving is a net gain by count and a real loss by coverage."""
    results = _results(["tests/a.py::test_alpha", "tests/a.py::test_gamma", "tests/a.py::test_delta"])
    problems = td.compare_to_baseline(results, _baseline_for(2), _MATRIX)
    assert any("STOPPED killing" in p for p in problems), problems


def test_identical_killer_sets_report_nothing(td):
    """Control: the new check must be silent when nothing moved, or it is noise."""
    results = _results(["tests/a.py::test_alpha", "tests/a.py::test_beta"])
    assert td.compare_to_baseline(results, _baseline_for(2), _MATRIX) == []


def test_new_killers_alone_are_reported_as_an_improvement_to_record(td):
    """Gains trip it too, same contract as the counts -- otherwise the next baseline
    encodes the gain as if it had always held."""
    results = _results(["tests/a.py::test_alpha", "tests/a.py::test_beta", "tests/a.py::test_gamma"])
    problems = td.compare_to_baseline(results, _baseline_for(2), _MATRIX)
    assert any("IMPROVED" in p for p in problems), problems


def test_an_ineffective_mutation_contributes_no_killer_noise(td):
    """Its zeros mean nothing, and the count ratchet already reports it.

    Without this the same mutation would be reported twice, once truthfully and once as a
    mass departure that is really just 'the mutation stopped applying'.
    """
    results = _results([], status="INEFFECTIVE")
    problems = td.compare_to_baseline(results, _baseline_for(2), _MATRIX)
    assert any("INEFFECTIVE" in p for p in problems)
    assert not any("STOPPED killing" in p for p in problems), problems


def test_without_a_matrix_the_comparison_itself_is_silent(td):
    """The function degrades cleanly; `main()` is what refuses, and on a distinct code.

    Split deliberately: `compare_to_baseline(..., None)` returning [] is the library
    contract, while "do not let a half-off gate exit 0" is a policy that belongs to the
    caller -- see `test_the_degraded_gate_cannot_exit_zero`.
    """
    results = _results(["tests/a.py::test_alpha", "tests/a.py::test_gamma"])
    assert td.compare_to_baseline(results, _baseline_for(2), None) == []
    assert "discrimination_killmatrix.json" in inspect.getsource(td.main), "main() no longer loads the matrix"


def test_the_reader_uses_the_key_main_writes(td):
    """The blocker in #1903's first revision, and no value-level test could see it.

    `_compare_killers` read `now["failed"]` while `main()` writes `"killed"`, so `after` was
    empty on every real run and all 220 baseline killers reported as departed on an unchanged
    tree. The six new tests passed because their fixture fabricated `"failed"` too.

    Asserted against the source of both sides, not against a fixture -- a fixture is what
    failed here.
    """
    writer = inspect.getsource(td.main)
    reader = inspect.getsource(td._compare_killers)
    assert '"killed": sorted(run.failed)' in writer, "main() no longer writes the killer list as 'killed'"
    assert 'now["killed"]' in reader, "the killer-set reader is not reading the key main() writes"
    assert 'now.get("failed"' not in reader, "the reader still reads the results dict under 'failed'"
    assert 'now["failed"]' not in reader, "the reader still reads the results dict under 'failed'"


def test_end_to_end_an_unchanged_tree_reports_nothing(td):
    """The integration the unit fixtures cannot give: real writer shape into real reader.

    Replays the committed matrix as if the sweep had just produced it. Anything other than
    silence means the two halves disagree about their own data, which is what shipped.
    """
    matrix = json.loads((_BASELINE.parent / "discrimination_killmatrix.json").read_text())
    baseline = json.loads(_BASELINE.read_text())
    results = {
        name: {
            "owner": entry.get("owner", "x"),
            "status": entry["status"],
            "killed": list(entry["killed"]),
            "kill_count": entry["kill_count"],
        }
        for name, entry in matrix["mutations"].items()
    }
    assert td.compare_to_baseline(results, baseline, matrix) == [], (
        "an unchanged tree is not silent: the reader and the writer disagree about their shape"
    )


def test_a_baseline_mutation_absent_from_the_matrix_is_reported_not_skipped(td):
    """Silently unchecked is class 2 of #1901 -- a verdict with no denominator."""
    results = _results(["tests/a.py::test_alpha", "tests/a.py::test_gamma"])
    results["a_mutation_the_matrix_never_saw"] = {
        "status": "ok",
        "kill_count": 1,
        "killed": ["tests/b.py::test_x"],
        "owner": "x",
    }
    baseline = _baseline_for(2)
    baseline["mutations"]["a_mutation_the_matrix_never_saw"] = {"kill_count": 1, "status": "ok", "owner": "x"}
    problems = td.compare_to_baseline(results, baseline, _MATRIX)
    assert any("NO matrix entry for" in p and "a_mutation_the_matrix_never_saw" in p for p in problems), problems
    assert any("compared for 1 of 2" in p for p in problems), problems


def test_a_rename_reports_the_arrival_too(td):
    """Departure alone says 'investigate'; departure WITH an arrival says 'probably a rename'.

    Suppressing arrivals whenever anything departed hid that distinction on exactly the case
    the design calls its intended cost.
    """
    results = _results(["tests/a.py::test_alpha", "tests/a.py::test_beta_renamed"])
    problems = td.compare_to_baseline(results, _baseline_for(2), _MATRIX)
    assert any("STOPPED killing" in p for p in problems), problems
    assert any("a rename? regenerate" in p for p in problems), problems


def test_the_degraded_gate_cannot_exit_zero(td):
    """A green run with half the gate off is invisible in a three-hour log."""
    body = inspect.getsource(td.main)
    assert "CANNOT MEASURE" in body, "the absent-matrix path no longer announces that it cannot measure"
    assert "sys.exit(2)" in body, "the absent-matrix path no longer exits on a code distinct from 0 and 1"


def test_the_regeneration_instruction_regenerates_both(td):
    """--write-baseline alone leaves the matrix stale, reddens the pinning test, and costs a
    second ~3h sweep."""
    body = inspect.getsource(td.main)
    assert "--json scripts/discrimination_killmatrix.json" in body, (
        "the failure text no longer tells the operator to regenerate the matrix too"
    )


def test_the_failure_parser_keeps_whole_node_ids(td):
    """A parametrisation label can contain spaces; `\\S+?` cut the ID at the first one.

    Harmless while both sides truncated identically. Not harmless once these strings are
    compared as identities: three params sharing a prefix collapse into one set member, and
    a kill_count of 1 stands for 3 distinct failing tests. Found by review (#1903).
    """
    spaced = (
        "FAILED tests/unit/test_alg/test_hjb_howard_solver.py::"
        "test_integrated_howard_matches_newton_nonlq[_v_quadratic-<lambda>-2.0-V+f(m), lambda=2]"
        " - AssertionError: boom"
    )
    assert td._FAILED.findall(spaced) == [
        "tests/unit/test_alg/test_hjb_howard_solver.py::"
        "test_integrated_howard_matches_newton_nonlq[_v_quadratic-<lambda>-2.0-V+f(m), lambda=2]"
    ]
    # Control: three params sharing a prefix must stay three, not collapse to one.
    trio = "\n".join(f"FAILED tests/a.py::test_p[x, case={n}] - AssertionError" for n in ("one", "two", "three"))
    assert len(set(td._FAILED.findall(trio))) == 3, "distinct parametrisations still collapse"
    # And the ordinary forms are untouched.
    assert td._FAILED.findall("FAILED tests/a.py::test_plain") == ["tests/a.py::test_plain"]
    assert td._FAILED.findall("ERROR tests/a.py::test_e - boom") == ["tests/a.py::test_e"]


def test_no_committed_killer_id_is_truncated(td):
    """The matrix is the gate's identity table now; a truncated entry is a silent collision."""
    matrix = json.loads((_BASELINE.parent / "discrimination_killmatrix.json").read_text())
    bad = [t for v in matrix["mutations"].values() for t in v.get("killed", ()) if t.count("[") != t.count("]")]
    assert not bad, f"truncated killer IDs in the committed matrix: {bad}"
