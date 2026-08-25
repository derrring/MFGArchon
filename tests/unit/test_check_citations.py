"""`scripts/check_citations.py` must see a drifted citation, and must not see one everywhere.

Issue #2102. The script measures `path.py:NNN` citations in prose whose named symbol is no longer
near the cited line. Two ways it can fail while reporting a number:

- **Blind.** A resolver or regex that stops matching returns zero drifted citations, which reads
  exactly like clean prose. Its own `--self-test` is the positive control for that; this file asserts
  the self-test still passes AND checks the categories independently, so a self-test that itself goes
  inert cannot hide behind a green run.
- **Indiscriminate.** A checker that reports every citation as drifted is as useless as one that
  reports none, and it would pass any test that only looks for findings. The `anchored` cases here
  are the negative control.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_citations.py"


def _git_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    """A real git repository: the script reads `git ls-files`, so an untracked tree measures zero."""
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    return tmp_path


def _measure(root: Path) -> dict:
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import importlib

        mod = importlib.import_module("check_citations")
        importlib.reload(mod)
        return mod.measure(root)
    finally:
        sys.path.pop(0)


TARGET = "\n".join(
    ["# header"]
    + [f"# filler {n}" for n in range(2, 30)]
    + ["def near_the_citation():", "    pass"]
    + [f"# filler {n}" for n in range(32, 120)]
    + ["def far_from_the_citation():", "    pass"]
)


def test_its_own_self_test_passes():
    """If this fails, nothing else in this file means anything: the instrument cannot see its own
    planted defects, so its verdict on the repository is unfounded."""
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test"], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "every category fires" in out.stdout


def test_a_symbol_near_the_cited_line_is_anchored(tmp_path):
    """The negative control. Without it, a checker that flags everything would pass this file."""
    root = _git_tree(
        tmp_path,
        {
            "pkg/target.py": TARGET,
            "doc.md": "`near_the_citation` is defined at pkg/target.py:30.\n",
        },
    )
    got = _measure(root)
    assert len(got["anchored"]) == 1, got
    assert got["drifted"] == []


def test_a_symbol_far_from_the_cited_line_is_drifted(tmp_path):
    root = _git_tree(
        tmp_path,
        {
            "pkg/target.py": TARGET,
            "doc.md": "`far_from_the_citation` is at pkg/target.py:30, the prose says.\n",
        },
    )
    got = _measure(root)
    assert len(got["drifted"]) == 1, got
    assert got["drifted"][0]["cited_line"] == 30


def test_a_citation_with_no_named_symbol_is_recorded_not_passed(tmp_path):
    """An unadjudicable row must never land in `anchored`. Counting it as clean is the failure this
    repository files under #1918: an unjudged row reads exactly like a passing one."""
    root = _git_tree(
        tmp_path,
        {"pkg/target.py": TARGET, "doc.md": "Something interesting happens at pkg/target.py:30.\n"},
    )
    got = _measure(root)
    assert len(got["unadjudicable"]) == 1, got
    assert got["anchored"] == []
    assert got["drifted"] == []


def test_the_symbol_walk_stops_at_a_blank_line(tmp_path):
    """A citation must not borrow a symbol from the next paragraph.

    This is the defect the script's self-test caught during development: with a flat +/-1 line
    window, seven unrelated claims on consecutive lines each anchored to a neighbour's symbol, and
    the drifted ones were reported clean.
    """
    doc = (
        "`near_the_citation` is a fine symbol.\n"
        "\n"
        "An unrelated remark about pkg/target.py:30.\n"
    )
    root = _git_tree(tmp_path, {"pkg/target.py": TARGET, "doc.md": doc})
    got = _measure(root)
    assert len(got["unadjudicable"]) == 1, f"the citation borrowed across a blank line: {got}"


def test_a_bare_basename_resolves_and_an_ambiguous_one_does_not(tmp_path):
    """Citations are usually written as a bare basename. Resolving the string as a path misses them
    -- the first version of this measurement reported 184 such citations as 'file does not exist',
    every one an artefact. Two files sharing a basename are reported ambiguous, never guessed."""
    root = _git_tree(
        tmp_path,
        {
            "pkg/deep/target.py": TARGET,
            "a/dup.py": "x = 1\n",
            "b/dup.py": "y = 2\n",
            "doc.md": "`near_the_citation` at target.py:30.\n\n`near_the_citation` at dup.py:1.\n",
        },
    )
    got = _measure(root)
    assert len(got["anchored"]) == 1, f"a bare basename did not resolve: {got}"
    assert len(got["ambiguous"]) == 1, f"an ambiguous basename was guessed: {got}"
    assert got["missing"] == []


def test_changelog_is_exempt_and_archive_is_skipped(tmp_path):
    """`CHANGELOG.md` describes released versions: an entry citing a v0.16 line is correct as of
    then. Same reasoning `scripts/check_doc_api.py` uses to exempt it."""
    cite = "`far_from_the_citation` at pkg/target.py:30.\n"
    root = _git_tree(
        tmp_path,
        {"pkg/target.py": TARGET, "CHANGELOG.md": cite, "archive/old.md": cite, "live.md": cite},
    )
    got = _measure(root)
    assert len(got["drifted"]) == 1, f"exempt prose was measured: {got}"
    assert got["drifted"][0]["file"] == "live.md"


def test_a_tree_that_is_not_a_repository_is_an_instrument_error(tmp_path):
    """`git ls-files` returning nothing must not be reported as 'no citations'. A silent zero from a
    failed query is the shape this checker exists to find in prose."""
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import importlib

        mod = importlib.import_module("check_citations")
        importlib.reload(mod)
        with pytest.raises(mod.InstrumentError):
            mod.measure(tmp_path)
    finally:
        sys.path.pop(0)
