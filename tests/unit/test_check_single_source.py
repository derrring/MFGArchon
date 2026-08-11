"""Discrimination tests for the single-source ratchet (scripts/check_single_source.py).

The ratchet's whole value is that a pattern which stops matching CANNOT read as clean
code. That failure is not hypothetical: on 2026-08-11 both ``% *Nx\\b`` and ``np\\.roll\\b``
returned 0 hits from ``git grep -E`` on this machine -- that grep does not implement
``\\b`` -- while the true counts were 18 and 18. Two candidate registry entries were a
keystroke away from being recorded as "already single-sourced".

The first version of this suite tested a weaker sentinel -- a literal string stored in the
registry that the pattern had to match -- and adversarial review broke it on 2026-08-12.
A literal proves the pattern intersects one hardcoded string, never that it still
describes the TREE, so an entry whose pattern had four alternatives passed green with
three of them structurally unreachable and 12 real sites uncounted. Hence
``test_respelling_a_site_cannot_launder_the_entry``: the sentinel is now a live site in
the file that OWNS the quantity, and drift that hides the tree from the pattern must exit
2 rather than print SHRANK and offer ``--write-baseline`` as the remedy.

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

    One of the counted sites is the owner's own return, which is what makes `owner.py` a
    usable sentinel file. Three further occurrences are decoys the matcher must not see: a
    docstring, a comment, and an f-string body. The f-string is not decoration -- on Python
    3.12+ it tokenizes as FSTRING_MIDDLE rather than STRING, so blanking only
    `tokenize.STRING` leaves its text visible, which is how a log message in
    `pde_coefficients.py:330` was counted as a site.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "owner.py").write_text(
        '"""Converter module.\n\nNever inline 0.5 * sigma * sigma in a solver.\n"""\n\n\n'
        "def diffusion(sigma):\n    return 0.5 * sigma * sigma\n"
    )
    (pkg / "solver.py").write_text(
        "def step(sigma, u):\n"
        "    # avoid writing 0.5 * sigma * sigma here\n"
        "    d = 0.5 * sigma * sigma\n"
        '    print(f"used 0.5 * sigma * sigma for {u}")\n'
        "    return d * u\n"
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


def test_comments_docstrings_and_fstrings_are_not_counted(tree, baseline):
    """Three of the five textual occurrences are prose; a raw-text scan would report 5."""
    result = _run(tree, baseline, "--list")
    assert result.returncode == EXIT_OK
    assert "2 site(s)" in result.stdout
    assert "pkg/owner.py:8" in result.stdout
    assert "pkg/solver.py:3" in result.stdout
    assert "pkg/solver.py:2" not in result.stdout  # the comment
    assert "pkg/solver.py:4" not in result.stdout  # the f-string body (FSTRING_MIDDLE on 3.12+)


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
    occur; the mechanism under test -- the pattern must match inside the file that owns the
    quantity -- is dialect-independent and catches both.
    """
    _patch_baseline(baseline, pattern=r"0\.5 \* sigma \* NOTASYMBOL")
    result = _run(tree, baseline)
    assert result.returncode == EXIT_INSTRUMENT_BROKEN, result.stdout
    assert "INSTRUMENT BROKEN" in result.stdout
    assert "matches nothing in its sentinel file" in result.stdout


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


def test_a_pattern_blind_to_the_owner_cannot_report_a_count(tree, baseline):
    """A pattern live SOMEWHERE but blind to the owner is still a broken instrument.

    This is what a stored-literal sentinel could not see, and what shipped: the pattern
    matched real sites, so it looked alive, while missing the spelling the owner actually
    uses. It must not be allowed to report a number.
    """
    (tree / "pkg" / "extra.py").write_text("def other(sigma):\n    return 0.5 * sigma * sigma\n")
    (tree / "pkg" / "owner.py").write_text(
        "def diffusion(sigma):\n    return 0.5 * sigma**2\n"  # ruff-canonical; the pattern is blind to it
    )
    result = _run(tree, baseline)
    assert result.returncode == EXIT_INSTRUMENT_BROKEN, result.stdout
    assert "no longer describes the tree" in result.stdout


def test_respelling_a_site_cannot_launder_the_entry(tree, baseline):
    """Drift that hides the tree from the pattern must not be recorded as progress.

    Reproduced against the real registry on 2026-08-12: respelling six sites from
    `0.5 * sigma * sigma` to the ruff-canonical `0.5 * sigma**2` -- which deletes nothing --
    dropped the count 6 -> 0, printed SHRANK, and `--write-baseline` then recorded the entry
    as clean forever while every site still restated D. Both sentinels stayed green because
    neither was a live site.
    """
    for name in ("owner.py", "solver.py"):
        path = tree / "pkg" / name
        path.write_text(path.read_text().replace("0.5 * sigma * sigma", "0.5 * sigma**2"))

    result = _run(tree, baseline)
    assert result.returncode == EXIT_INSTRUMENT_BROKEN, result.stdout
    assert "SHRANK" not in result.stdout
    assert "--write-baseline" not in result.stdout

    # And the remedy the old version offered must not silently succeed either.
    laundered = _run(tree, baseline, "--write-baseline")
    assert laundered.returncode == EXIT_INSTRUMENT_BROKEN, laundered.stdout
    assert json.loads(baseline.read_text())["entries"][0]["count"] == 2


def test_dead_globs_are_instrument_error(tree, baseline):
    """Second instrument failure mode: the sentinel file is no longer scanned."""
    _patch_baseline(baseline, exclude=["pkg/owner.py"])
    result = _run(tree, baseline)
    assert result.returncode == EXIT_INSTRUMENT_BROKEN, result.stdout
    assert "is not among the" in result.stdout


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
