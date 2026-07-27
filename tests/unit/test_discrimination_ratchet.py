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


def test_the_kill_matrix_is_committed_beside_the_baseline(td):
    """The counts are the gate; the matrix is the evidence under them.

    Without it the population claims in #1715 -- "193 of 308 inert", "39 of 65" -- are
    unrecoverable by anyone, which is the shape of claim this whole campaign exists to
    stop shipping.
    """
    matrix = json.loads((_BASELINE.parent / "discrimination_killmatrix.json").read_text())
    assert matrix["_selection_regex_for_agreement_shaped"]
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
