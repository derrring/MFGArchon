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

import os
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


DRIFTED_LINE = "`far_from_the_citation` is at pkg/target.py:30, the prose says.\n"

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
    out = subprocess.run([sys.executable, str(SCRIPT), "--self-test"], capture_output=True, text=True, check=False)
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


def test_a_failing_unmerged_probe_refuses_instead_of_passing(tmp_path, monkeypatch):
    """The guard's returncode check. Without it a failing query returns empty stdout, the guard's
    `if` is false, and the tripled count it exists to refuse is handed back as a clean measurement
    with exit 0 -- which is the silent-zero-from-a-broken-query shape this whole script is about.

    The shim fails ONLY on `--unmerged`, so every other git call still works and the run reaches
    the guard rather than dying earlier for an unrelated reason.
    """
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "git").write_text(
        '#!/bin/sh\nfor a in "$@"; do [ "$a" = "--unmerged" ] && exit 3; done\nexec /usr/bin/git "$@"\n'
    )
    (shim / "git").chmod(0o755)
    root = _git_tree(tmp_path / "repo", {"doc.md": "`near_the_citation` at pkg/target.py:30.\n", "pkg/target.py": TARGET})
    monkeypatch.setenv("PATH", f"{shim}:{os.environ['PATH']}")
    out = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False, cwd=str(root)
    )
    assert out.returncode == 2, out.stdout + out.stderr
    assert "--unmerged failed" in out.stderr, out.stderr


def test_a_tracked_symlink_is_not_counted_a_second_time(tmp_path):
    """`CLAUDE.md` in this repository is a tracked symlink to `AGENTS.md`: one paragraph, two
    tracked paths, and its citations were counted twice -- the published `drifted` was one row high
    because of it, 19/39 where the truth is 18/38.

    No name-based rule finds this. The guard's first two shapes were casefold, then NFC plus
    casefold, and each was defeated by a case outside it; these two names share no case and no
    Unicode form. Only the filesystem knows they are one file, so the guard asks the filesystem.
    """
    root = _git_tree(tmp_path, {"doc.md": DRIFTED_LINE, "pkg/target.py": TARGET})
    (root / "link.md").symlink_to(root / "doc.md")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    mode = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-s", "link.md"], capture_output=True, text=True, check=True
    ).stdout
    assert mode.startswith("120000"), f"the fixture did not stage a symlink: {mode!r}"

    got = _measure(root)
    assert len(got["drifted"]) == 1, f"the symlinked copy was counted again: {got['drifted']}"
    assert {r["file"] for r in got["drifted"]} == {"doc.md"}


def test_two_NON_symlink_entries_on_one_inode_are_refused(tmp_path):
    """The other verdict, and the reason the guard does not simply skip every collision: case,
    Unicode form or a hard link all reach one file through two ORDINARY index entries, and that is
    a broken index rather than a legitimate alias."""
    root = _git_tree(tmp_path, {"doc.md": DRIFTED_LINE, "pkg/target.py": TARGET})
    if (root / "DOC.MD").exists() != (root / "doc.md").exists():
        pytest.skip("case-sensitive filesystem: `Doc.md` and `doc.md` really are two files here")
    git = ["git", "-C", str(root)]
    sha = subprocess.run(
        [*git, "hash-object", "-w", "doc.md"], capture_output=True, text=True, check=True
    ).stdout.strip()
    subprocess.run([*git, "update-index", "--add", "--cacheinfo", f"100644,{sha},Doc.md"], check=True)
    assert not subprocess.run(
        [*git, "ls-files", "--unmerged"], capture_output=True, text=True, check=True
    ).stdout.strip(), "this must not be an unmerged index -- that is the other guard"

    out = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False, cwd=str(root)
    )
    assert out.returncode == 2, out.stdout + out.stderr
    assert "at least two of them" in out.stderr, out.stderr


def _stage_symlink(root: Path, link: str, target: str) -> None:
    (root / link).symlink_to(target)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    mode = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-s", link], capture_output=True, text=True, check=True
    ).stdout
    assert mode.startswith("120000"), f"{link} was not staged as a symlink: {mode!r}"


def test_two_symlinks_to_an_UNTRACKED_target_are_not_refused(tmp_path):
    """A group whose members are ALL symlinks. The previous rule required exactly one non-symlink,
    so this legitimate tree was refused -- and told "none of them is a symlink" while both were.
    The rule counts regular files now instead of matching a group shape."""
    root = _git_tree(tmp_path, {"pkg/target.py": TARGET})
    (root / "untracked.md").write_text(DRIFTED_LINE)
    (root / ".gitignore").write_text("untracked.md\n")
    _stage_symlink(root, "s1.md", "untracked.md")
    _stage_symlink(root, "s2.md", "untracked.md")

    got = _measure(root)
    assert len(got["drifted"]) == 1, f"one file, one citation, counted {len(got['drifted'])} times"


def test_a_symlink_chain_collapses_to_one_count(tmp_path):
    """`link2 -> link1 -> real`, all three tracked: one file, one citation."""
    root = _git_tree(tmp_path, {"real.md": DRIFTED_LINE, "pkg/target.py": TARGET})
    _stage_symlink(root, "link1.md", "real.md")
    _stage_symlink(root, "link2.md", "link1.md")

    got = _measure(root)
    assert len(got["drifted"]) == 1, f"the chain was counted {len(got['drifted'])} times"
    assert got["drifted"][0]["file"] == "real.md", got["drifted"]


def test_the_regular_file_is_the_one_kept(tmp_path):
    """Which representative survives is a contract, not an accident: the citation must be reported
    against the real file, so that a reader following the report edits the file rather than a link.
    """
    root = _git_tree(tmp_path, {"real.md": DRIFTED_LINE, "pkg/target.py": TARGET})
    _stage_symlink(root, "aaa_sorts_first.md", "real.md")

    got = _measure(root)
    assert [r["file"] for r in got["drifted"]] == ["real.md"], got["drifted"]
# --- the ratchet (#2102 part 2) --------------------------------------------------------------
#
# Independent of `--self-test`, which exercises the same three shapes: a self-test asserting its own
# correctness is not evidence, and the ratchet is the half that can go inert without any category
# count changing.


def _ratchet(root: Path, baseline: Path) -> tuple[int, str]:
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--check-baseline", str(baseline)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(root),
    )
    return out.returncode, out.stdout + out.stderr


def _with_baseline(tmp_path: Path, doc: str) -> tuple[Path, Path]:
    root = _git_tree(tmp_path, {"pkg/target.py": TARGET, "doc.md": doc})
    baseline = tmp_path / "baseline.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--write-baseline", str(baseline)],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(root),
    )
    return root, baseline


DRIFTED = "`far_from_the_citation` is at pkg/target.py:30, the prose says.\n"


def test_an_unchanged_tree_passes_its_own_baseline(tmp_path):
    """The negative control for the ratchet. A gate that is red on arrival teaches everyone to
    pass `--no-verify`, and it would satisfy every other test in this section."""
    root, baseline = _with_baseline(tmp_path, DRIFTED)
    rc, out = _ratchet(root, baseline)
    assert rc == 0, out
    assert "citation ratchet OK" in out


def test_a_new_drifted_citation_to_an_ALREADY_RECORDED_line_fails(tmp_path):
    """Identities alone cannot see this one, which is why the count check stayed. Two sentences in
    one file citing the same target line collapse to one key, so the set does not grow -- the
    script's own `--self-test` caught that within a minute of the identity check being written."""
    root, baseline = _with_baseline(tmp_path, DRIFTED)
    (root / "doc.md").write_text(DRIFTED + "\n" + DRIFTED.replace("is at", "also at"))
    rc, out = _ratchet(root, baseline)
    assert rc == 1, out
    assert "with the same set of claims" in out


def test_a_new_drifted_citation_to_a_FRESH_line_names_it(tmp_path):
    """And counts alone cannot see a compensating pair -- hide two rows, add two, the total is
    unchanged. Measured on this repository, that shipped two new broken citations through a green
    gate. So the failure must name WHICH claim, not only how many."""
    root, baseline = _with_baseline(tmp_path, DRIFTED)
    (root / "doc.md").write_text(DRIFTED + "\n" + DRIFTED.replace(":30,", ":31,"))
    rc, out = _ratchet(root, baseline)
    assert rc == 1, out
    assert "no longer near the line they point at" in out
    assert "doc.md -> pkg/target.py:31" in out, "the failure must name WHICH citation"


def test_deleting_the_symbol_name_does_NOT_read_as_an_improvement(tmp_path):
    """The reason the ratchet pins two numbers. Dropping the backticked symbol moves the row out of
    the numerator AND the denominator, so a `drifted`-only ratchet records the cheapest possible
    evasion as progress.

    The message half is asserted too, and what it asserts changed after four review rounds: the
    branch no longer says WHY a row left. Every attempt to adjudicate that -- hidden versus fixed --
    accused a correct repair, or suppressed its own test shape, or made the evasion and its opposite
    byte-identical. It reports the rows and the reader decides, so what is pinned here is that the
    row is NAMED, not what it is called."""
    root, baseline = _with_baseline(tmp_path, DRIFTED)
    (root / "doc.md").write_text(DRIFTED.replace("`far_from_the_citation` is", "Something is"))
    rc, out = _ratchet(root, baseline)
    assert rc == 1, out
    assert "adjudicable 1 -> 0" in out
    assert "no longer drifted" in out
    assert "doc.md -> pkg/target.py:30" in out, "the row that left must be named, not just counted"


def test_a_fixed_citation_fails_until_the_baseline_records_it(tmp_path):
    """Bidirectional, matching `capability_matrix` and `check_single_source`: an unrecorded
    improvement is where the next regression hides, inside a number nobody re-read."""
    root, baseline = _with_baseline(tmp_path, DRIFTED)
    (root / "doc.md").write_text(DRIFTED.replace(":30,", ":120,"))
    rc, out = _ratchet(root, baseline)
    assert rc == 1, out
    assert "no longer drifted" in out


def test_correct_new_prose_passes(tmp_path):
    """Adding a citation that is right must not be refused -- otherwise the gate taxes writing
    documentation, and the cheapest way to stay green becomes citing nothing."""
    root, baseline = _with_baseline(tmp_path, DRIFTED)
    (root / "doc.md").write_text(DRIFTED + "\n`near_the_citation` is at pkg/target.py:30.\n")
    rc, out = _ratchet(root, baseline)
    assert rc == 0, out


def test_a_missing_baseline_is_reported_not_silently_passed(tmp_path):
    root = _git_tree(tmp_path, {"pkg/target.py": TARGET, "doc.md": DRIFTED})
    rc, out = _ratchet(root, tmp_path / "no_such_baseline.json")
    assert rc == 2, out
    assert "CANNOT COMPARE" in out


def test_the_shipped_baseline_matches_this_repository():
    """A baseline written from an uncommitted tree records a commit that does not describe what was
    measured. This fails the moment the repository's own citations move without the baseline."""
    repo = SCRIPT.parents[1]
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--check-baseline"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(repo),
    )
    assert out.returncode == 0, out.stdout + out.stderr


def _commit_tree(tmp_path: Path, files: dict[str, str]) -> tuple[Path, list[str]]:
    root = _git_tree(tmp_path, files)
    git = ["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    return root, git


def _write_baseline_at(root: Path, baseline: Path) -> str:
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import importlib

        mod = importlib.reload(importlib.import_module("check_citations"))
        mod.write_baseline(mod.measure(root), baseline, root)
    finally:
        sys.path.pop(0)
    import json

    return json.loads(baseline.read_text())["_measured_at"]["head_when_written"]


def test_a_baseline_from_a_dirty_tree_says_so(tmp_path):
    """`-dirty` exists so the next reader knows the recorded commit does not describe what was
    measured. Removing the suffix entirely was killed by zero tests before this one."""
    root, _ = _commit_tree(tmp_path, {"doc.md": DRIFTED, "pkg/target.py": TARGET})
    (root / "doc.md").write_text(DRIFTED + "\nan unrelated edit\n")
    assert _write_baseline_at(root, tmp_path / "b.json").endswith("-dirty")


def test_the_baseline_file_itself_does_not_make_the_tree_dirty(tmp_path):
    """The carve-out, and the reason it is not cosmetic: writing the baseline modifies the baseline,
    so without it EVERY baseline is stamped `-dirty` including one written from a clean tree, and a
    marker that is always on discriminates nothing.

    This is also the regression test for how it was broken: `git status --porcelain` emits
    `XY PATH`, an unstaged modification starts with a SPACE, and `.stdout.strip()` ate that space
    off the first line only -- shifting the path by one character so the comparison never matched.
    Removing the carve-out was likewise killed by zero tests.
    """
    root, git = _commit_tree(tmp_path, {"doc.md": DRIFTED, "pkg/target.py": TARGET})
    baseline = root / "citation_baseline.json"
    # TRACKED and committed first, then modified. An untracked baseline reports `?? path`, whose
    # first character is not a space -- and the bug this pins only bites on ` M path`, so a fixture
    # left untracked passes with the defect restored. Measured: it did, over all 26 tests.
    _write_baseline_at(root, baseline)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "record baseline"], check=True)
    _write_baseline_at(root, baseline)

    porcelain = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    assert porcelain.startswith(" M "), (
        f"this fixture must leave the baseline TRACKED and UNSTAGED, or it cannot exhibit the "
        f"defect it pins: {porcelain!r}"
    )
    assert not _write_baseline_at(root, baseline).endswith("-dirty")


def test_rewriting_a_bare_basename_to_its_full_path_is_the_same_claim(tmp_path):
    """The identity key resolves the target before keying on it. Keying on the literal citation
    text reported one unchanged claim, written more precisely, as a regression AND an improvement --
    and that rewrite is exactly the correct repair for the `ambiguous` and `missing` rows."""
    root, baseline = _with_baseline(tmp_path, "`far_from_the_citation` is at target.py:30.\n")
    (root / "doc.md").write_text("`far_from_the_citation` is at pkg/target.py:30.\n")
    rc, out = _ratchet(root, baseline)
    assert rc == 0, out


def test_a_baseline_written_OUTSIDE_the_repository_does_not_crash(tmp_path):
    """`--write-baseline /tmp/x.json` is a legitimate invocation and it raised ValueError: the
    dirty-check's carve-out called `path.relative_to(root)` unguarded. Found by a replay harness
    whose positive control went red, not by reading the code."""
    root, _ = _commit_tree(tmp_path / "repo", {"doc.md": DRIFTED, "pkg/target.py": TARGET})
    outside = tmp_path / "outside.json"
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--write-baseline", str(outside)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(root),
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert outside.is_file(), "the baseline was not written"
