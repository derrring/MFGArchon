"""The frozen-area ratchet must fail in BOTH directions, and something must check that it does.

`check_frozen_areas.py --self-test` covers `_references`: every branch that turns a source file
into a detection has a fixture, and deleting a branch reddens the gate naming it. It stops there.
The code that turns a detection into a *failure* -- the two set differences in `main()` -- had no
automated control at all, which is the same defect one level up: three single-line mutations left
`--check-baseline` at exit 0 while `--self-test` printed PASSED.

    sorted(set(found[p]) - recorded[p]) -> []   an added frozen test passes
    return 1 inside `if added:`         -> 0    an added frozen test passes
    sorted(recorded[p] - set(found[p])) -> []   a deleted frozen test passes

The third falsifies the property the ratchet is advertised on -- hard-failing on a drop, the way
`check_doc_api.py` does and `check_fail_fast.py` does not.

Shaped after `tests/unit/test_check_fail_fast.py`, which pins its sibling ratchet's comparison for
the same reason. Both directions are exercised against a synthetic tree, so the assertions do not
depend on what the real baseline happens to contain today.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "check_frozen_areas.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("check_frozen_areas", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tests_dir: pathlib.Path, baseline: pathlib.Path) -> subprocess.CompletedProcess:
    """The gate as CI invokes it: a subprocess, so the exit code is the thing measured."""
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--tests", str(tests_dir), "--check-baseline", str(baseline)],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def frozen_tree(tmp_path, script):
    """A two-file synthetic tree at baseline: one frozen-area test per frozen package."""
    tests_dir = tmp_path / "tests"
    (tests_dir / "unit").mkdir(parents=True)
    (tests_dir / "unit" / "test_neural_pinned.py").write_text("import mfgarchon.alg.neural.nn\n")
    (tests_dir / "unit" / "test_rl_pinned.py").write_text("from mfgarchon.alg.reinforcement.algorithms import x\n")

    found = script.offending_files(tests_dir)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"counts": {p: len(f) for p, f in found.items()}, "files": found}) + "\n")
    return tests_dir, baseline


def test_the_tree_starts_green(frozen_tree):
    """Without this, every assertion below could be satisfied by a gate that always fails."""
    tests_dir, baseline = frozen_tree
    result = _run(tests_dir, baseline)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: no new tests against frozen paradigms" in result.stdout


def test_an_added_frozen_test_fails_the_gate_and_is_named(frozen_tree):
    tests_dir, baseline = frozen_tree
    added = tests_dir / "unit" / "test_rl_new.py"
    added.write_text("from mfgarchon.alg.reinforcement.environments import Env\n")

    result = _run(tests_dir, baseline)
    assert result.returncode == 1, "an added frozen-area test must fail the gate\n" + result.stdout
    assert "new tests added against a FROZEN prototype paradigm" in result.stdout
    assert "test_rl_new.py" in result.stdout, "the message must name the file, not just the count"
    assert "--write-baseline" in result.stdout, "and say how to proceed deliberately"


def test_a_deleted_frozen_test_fails_the_gate(frozen_tree):
    """The drop direction. `check_fail_fast.py` prints a nudge and exits 0 here; this must not.

    A silent drop means the baseline no longer describes the tree, so the next addition is measured
    against a stale record -- and delete-one-add-one nets to zero under a count comparison, which
    is the shape this gate uses sets to catch.
    """
    tests_dir, baseline = frozen_tree
    (tests_dir / "unit" / "test_neural_pinned.py").unlink()

    result = _run(tests_dir, baseline)
    assert result.returncode == 1, "a removed frozen-area test must fail the gate\n" + result.stdout
    assert "disappeared" in result.stdout


def test_delete_one_add_one_does_not_net_to_zero(frozen_tree, script):
    """The case that motivated comparing sets rather than counts.

    Counts are unchanged here -- one neural file out, one in -- so a count comparison reports no
    change and the substituted test is never seen.
    """
    tests_dir, baseline = frozen_tree
    (tests_dir / "unit" / "test_neural_pinned.py").unlink()
    (tests_dir / "unit" / "test_neural_substituted.py").write_text("import mfgarchon.alg.neural.nn\n")

    before = json.loads(baseline.read_text())["counts"]
    after = {p: len(f) for p, f in script.offending_files(tests_dir).items()}
    assert before == after, f"the fixture must hold the counts equal, or it tests nothing: {before} vs {after}"

    result = _run(tests_dir, baseline)
    assert result.returncode == 1, "a substituted frozen test must fail the gate\n" + result.stdout
    assert "test_neural_substituted.py" in result.stdout


def test_a_non_frozen_test_does_not_fail_the_gate(frozen_tree):
    """Negative control: the gate must be silent about the rest of the suite.

    Without this, every assertion above is satisfied by a gate that fails on any new file.
    """
    tests_dir, baseline = frozen_tree
    (tests_dir / "unit" / "test_ordinary.py").write_text(
        "from mfgarchon.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver\n"
    )

    result = _run(tests_dir, baseline)
    assert result.returncode == 0, "a non-frozen test must not trip the gate\n" + result.stdout
