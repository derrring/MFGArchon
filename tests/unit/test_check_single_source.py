"""Discrimination tests for the single-source ratchet (scripts/check_single_source.py).

The ratchet's whole value is that a pattern which stops matching CANNOT read as clean
code. That failure is not hypothetical: on 2026-08-11 both ``% *Nx\\b`` and ``np\\.roll\\b``
returned 0 hits from ``git grep -E`` on this machine -- that grep does not implement
``\\b`` -- while the true counts were 18 and 18. Two candidate registry entries were a
keystroke away from being recorded as "already single-sourced".

So the load-bearing test here is ``test_dead_pattern_is_instrument_error``: break the
pattern and the check must exit 2, never 0 (silently clean) and never 1 (a verdict about
the tree). The rest pin the both-directions ratchet and the second instrument failure
mode, dead include/exclude globs.

Every test builds its own synthetic tree and registry, so none of them depends on the
real baseline's current counts -- except ``test_repo_baseline_is_current``, which is the
one that must track the tree.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "check_single_source.py"

EXIT_OK = 0
EXIT_COUNT_CHANGED = 1
EXIT_INSTRUMENT_BROKEN = 2


def _run(root: Path, baseline: Path, *extra: str) -> subprocess.CompletedProcess:
    """Invoke the checker as the gate does -- exit code is the contract local_ci.sh reads."""
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--root", str(root), "--baseline", str(baseline), *extra],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Synthetic package with exactly two restatements of `0.5 * sigma * sigma`.

    The third occurrence is inside a docstring and the fourth inside a comment: both must
    be invisible, which is what forces tokenize-based blanking rather than raw text.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "owner.py").write_text(
        '"""Converter module.\n\nNever inline 0.5 * sigma * sigma in a solver.\n"""\n\n\ndef diffusion(sigma):\n    return 0.5 * sigma * sigma\n'
    )
    (pkg / "solver.py").write_text(
        "def step(sigma, u):\n    # avoid writing 0.5 * sigma * sigma here\n    d = 0.5 * sigma * sigma\n    return d * u\n"
    )
    return tmp_path


@pytest.fixture
def baseline(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "diffusion",
                        "quantity": "D = sigma^2 / 2",
                        "owner": "pkg/owner.py::diffusion",
                        "pattern": r"0\.5 \* sigma \* sigma",
                        "include": ["pkg/**/*.py"],
                        "exclude": [],
                        "sentinel_text": "    return 0.5 * sigma * sigma",
                        "sentinel_file": "pkg/owner.py",
                        "count": 2,
                        "note": "synthetic",
                    }
                ]
            }
        )
    )
    return path


def _patch_baseline(baseline: Path, **fields) -> None:
    data = json.loads(baseline.read_text())
    data["entries"][0].update(fields)
    baseline.write_text(json.dumps(data))


def test_clean_tree_at_baseline_passes(tree, baseline):
    result = _run(tree, baseline)
    assert result.returncode == EXIT_OK, result.stdout


def test_comments_and_docstrings_are_not_counted(tree, baseline):
    """Two of the four textual occurrences are prose; a raw-text scan would report 4."""
    result = _run(tree, baseline, "--list")
    assert result.returncode == EXIT_OK
    assert "2 site(s)" in result.stdout
    assert "pkg/owner.py:8" in result.stdout
    assert "pkg/solver.py:3" in result.stdout
    assert "pkg/solver.py:2" not in result.stdout  # the comment


def test_new_restatement_turns_it_red(tree, baseline):
    """Growth is a regression: a third site must fail the gate."""
    (tree / "pkg" / "extra.py").write_text("def other(sigma):\n    return 0.5 * sigma * sigma\n")
    result = _run(tree, baseline)
    assert result.returncode == EXIT_COUNT_CHANGED, result.stdout
    assert "GREW" in result.stdout
    assert "pkg/extra.py:2" in result.stdout


def test_removed_restatement_also_turns_it_red(tree, baseline):
    """Shrink is progress that must be recorded -- silently accepting it loses the claim."""
    (tree / "pkg" / "solver.py").write_text("def step(sigma, u):\n    return diffusion(sigma) * u\n")
    result = _run(tree, baseline)
    assert result.returncode == EXIT_COUNT_CHANGED, result.stdout
    assert "SHRANK" in result.stdout
    assert "--write-baseline" in result.stdout


def test_dead_pattern_is_instrument_error(tree, baseline):
    """A pattern that matches nothing must exit 2, not report a clean tree.

    ``\\b`` is the exact keystroke that produced two false zeros on 2026-08-11. Python's
    ``re`` does implement it, so the synthetic dead pattern here is a literal that cannot
    occur; the mechanism under test -- the pattern must match its own sentinel text -- is
    dialect-independent and catches both.
    """
    _patch_baseline(baseline, pattern=r"0\.5 \* sigma \* NOTASYMBOL")
    result = _run(tree, baseline)
    assert result.returncode == EXIT_INSTRUMENT_BROKEN, result.stdout
    assert "INSTRUMENT BROKEN" in result.stdout
    assert "sentinel text" in result.stdout


def test_dead_pattern_is_not_reported_as_progress(tree, baseline):
    """The failure mode this ratchet exists to prevent, stated as its own assertion.

    Without the sentinel, a dead pattern counts 0 against a baseline of 2 and prints
    SHRANK -- an instrument failure dressed as a fix, with `--write-baseline` offered as
    the remedy. Asserting the absence of that text pins the distinction, not just the code.
    """
    _patch_baseline(baseline, pattern=r"0\.5 \* sigma \* NOTASYMBOL")
    result = _run(tree, baseline)
    assert "SHRANK" not in result.stdout
    assert "OK:" not in result.stdout


def test_dead_globs_are_instrument_error(tree, baseline):
    """Second instrument failure mode: the sentinel file is no longer scanned."""
    _patch_baseline(baseline, exclude=["pkg/owner.py"])
    result = _run(tree, baseline)
    assert result.returncode == EXIT_INSTRUMENT_BROKEN, result.stdout
    assert "sentinel file" in result.stdout


def test_globs_selecting_nothing_is_instrument_error(tree, baseline):
    _patch_baseline(baseline, include=["nosuchdir/**/*.py"])
    result = _run(tree, baseline)
    assert result.returncode == EXIT_INSTRUMENT_BROKEN, result.stdout
    assert "selected no files" in result.stdout


def test_untokenizable_file_is_instrument_error(tree, baseline):
    """A file the scanner cannot read is an instrument failure, not a count of 2.

    Letting `TokenError` escape would exit 1, and exit 1 is the gate's "the count changed"
    verdict -- a crash would be reported as a fact about the tree.
    """
    (tree / "pkg" / "broken.py").write_text("def f(:\n    '''unterminated\n")
    result = _run(tree, baseline)
    assert result.returncode == EXIT_INSTRUMENT_BROKEN, result.stdout
    assert "cannot tokenize" in result.stdout


def test_write_baseline_records_the_new_count(tree, baseline):
    (tree / "pkg" / "extra.py").write_text("def other(sigma):\n    return 0.5 * sigma * sigma\n")
    assert _run(tree, baseline, "--write-baseline").returncode == EXIT_OK
    assert json.loads(baseline.read_text())["entries"][0]["count"] == 3
    assert _run(tree, baseline).returncode == EXIT_OK


def test_repo_baseline_is_current():
    """The real registry against the real tree -- what local_ci.sh runs."""
    result = _run(_REPO, _REPO / "scripts" / "single_source_baseline.json")
    assert result.returncode == EXIT_OK, result.stdout


def test_every_repo_entry_names_an_owner_or_says_there_is_none():
    """An entry with no stated owner is a count with no meaning attached to it."""
    registry = json.loads((_REPO / "scripts" / "single_source_baseline.json").read_text())
    assert registry["entries"], "registry is empty"
    for entry in registry["entries"]:
        assert entry["owner"].strip(), f"{entry['name']}: empty owner"
        assert entry["note"].strip(), f"{entry['name']}: empty note"
