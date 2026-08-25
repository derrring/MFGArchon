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


def test_a_citation_cannot_borrow_a_symbol_from_an_ADJACENT_line(tmp_path):
    """The citation's own line must name what it cites. Nothing is borrowed from a neighbour.

    This replaces `test_the_symbol_walk_stops_at_a_blank_line`, which was tautological: it separated
    the symbol from the citation with a BLANK line, and a blank line holds no backtick, so it passed
    with the guard and without it. The guard it named was inert and deleting it killed nothing.

    The line below is the discriminator that one was not -- a non-blank neighbour carrying a real
    symbol, which the previous walk would have borrowed. Measured on this repository, that borrowing
    was live: 20 of 41 anchored rows anchored ONLY on a symbol absent from their own line.
    """
    doc = "`near_the_citation` is a fine symbol.\nAn unrelated remark about pkg/target.py:30.\n"
    root = _git_tree(tmp_path, {"pkg/target.py": TARGET, "doc.md": doc})
    got = _measure(root)
    assert len(got["unadjudicable"]) == 1, f"the citation borrowed from its neighbour: {got}"
    assert not got["anchored"], got


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


# --- the blockers from lane `citation-measure` -------------------------------------------------


def test_a_citation_past_EOF_is_drifted_even_with_NO_symbol_named(tmp_path):
    """Gated on a symbol, as it was first written, the ONE certainly-broken citation in this
    repository was filed `unadjudicable` -- and the module docstring rested its central argument on
    a finding the shipped instrument did not report. The real case: `geometry/boundary/protocols.py`
    cites line 2664 of `applicator_fdm.py`, a file with 2182 lines.

    Written that way on purpose. Spelled as a citation, this docstring BECOMES another instance of
    the defect it describes -- measured, it landed in `drifted` and moved the number the ratchet
    pins. Prose quoting a broken citation as evidence is still prose this script scans."""
    root = _git_tree(
        tmp_path, {"pkg/target.py": TARGET, "doc.md": "Nothing named, yet pkg/target.py:99999.\n"}
    )
    got = _measure(root)
    assert len(got["drifted"]) == 1, got
    assert not got["unadjudicable"], got
    assert "past EOF" in got["drifted"][0]["why"]


def test_a_tracked_but_deleted_target_is_reported_not_a_traceback(tmp_path):
    """`git ls-files` lists a file deleted from the working tree, so the resolver hands back a path
    that does not open. Unwrapped this was a traceback and exit 1 from a script whose whole contract
    is that it exits 0 whatever it finds -- and it becomes load-bearing once this runs in the gate,
    where any developer mid-`rm` would get it."""
    root = _git_tree(
        tmp_path,
        {"pkg/gone.py": "x = 1\n", "doc.md": "`near_the_citation` is at pkg/gone.py:1.\n"},
    )
    (root / "pkg" / "gone.py").unlink()
    out = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False, cwd=str(root)
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Traceback" not in out.stderr, out.stderr


def test_a_citation_cannot_read_a_file_outside_the_repository(tmp_path):
    """`root / rel` was not confined: `sub/../../outside/escape.py:3` read a file outside the tree
    and reported it anchored."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "escape.py").write_text("def near_the_citation():\n    pass\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    root = _git_tree(repo, {"sub/doc.md": "`near_the_citation` at sub/../../outside/escape.py:1.\n"})
    got = _measure(root)
    assert not got["anchored"], got


def test_the_self_test_notices_when_an_exemption_is_dropped():
    """A negative control on the self-test itself. Emptying `EXEMPT_DIRS` or dropping the CHANGELOG
    exemption both SURVIVED it as first written -- it had no shape for either branch, so those two
    were pinned by this file alone and nothing said the instrument's own control was thinner."""
    import importlib

    sys.path.insert(0, str(SCRIPT.parent))
    try:
        mod = importlib.reload(importlib.import_module("check_citations"))
        for attr, value in (("EXEMPT_DIRS", set()), ("EXEMPT_FILES", set())):
            keep = getattr(mod, attr)
            setattr(mod, attr, value)
            try:
                assert mod.self_test(Path.cwd()) == 1, f"self-test survived {attr} = {value!r}"
            finally:
                setattr(mod, attr, keep)
    finally:
        sys.path.pop(0)


def test_a_tree_mid_conflict_is_refused_not_counted_three_times(tmp_path):
    """`git ls-files` lists an unmerged path once per stage, so every citation in a conflicted file
    is counted three times -- found by rebasing this very branch, where `missing` read 62 against
    the resolved tree's 30. Both halves are asserted: no duplication, and no verdict at all over a
    tree holding `<<<<<<<` markers and both versions of every line."""
    git = ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t"]
    root = _git_tree(tmp_path, {"doc.md": "`near_the_citation` at pkg/target.py:30.\n", "pkg/target.py": TARGET})
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    base = subprocess.run(
        [*git, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    subprocess.run([*git, "checkout", "-qb", "other"], check=True)
    (root / "doc.md").write_text("`far_from_the_citation` at pkg/target.py:30.\n")
    subprocess.run([*git, "commit", "-aqm", "theirs"], check=True)
    subprocess.run([*git, "checkout", "-q", base], check=True)
    (root / "doc.md").write_text("`near_the_citation` at pkg/target.py:31.\n")
    subprocess.run([*git, "commit", "-aqm", "ours"], check=True)
    subprocess.run([*git, "merge", "other"], capture_output=True, check=False)

    assert subprocess.run(
        [*git, "ls-files", "--unmerged"], capture_output=True, text=True, check=True
    ).stdout.strip(), "the fixture did not actually produce a conflict -- this test proves nothing"

    out = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False, cwd=str(root)
    )
    assert out.returncode == 2, out.stdout + out.stderr
    assert "unmerged" in out.stderr, out.stderr
