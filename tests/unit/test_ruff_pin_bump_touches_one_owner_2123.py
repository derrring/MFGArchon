"""#2123: the ruff bump must touch the ruff pin, nothing else, and never silently do nothing.

`.pre-commit-config.yaml` carries two `rev:` lines -- `astral-sh/ruff-pre-commit` and
`pre-commit/pre-commit-hooks`. The monthly workflow's `sed` was unanchored, and `s///` applies once
PER LINE, so a bump set `pre-commit-hooks` to the ruff version: a tag that does not exist. Measured
by the reviewer against real `pre-commit` 4.6.1 -- `pre-commit run --all-files` exits 3 and the
healthy hook, listed first, never runs; naming only the healthy hook fails too, because it dies
during repo initialization. `git push` is then blocked for every contributor. And no workflow runs
`pre-commit`, so the bot's own PR checks would have been GREEN.

The workflow half cannot be unit-tested from here -- it is a `sed` in an Actions step -- and carries
its own guard, which re-reads the pin the way `ci.yml` does rather than counting lines. This file
pins the reachable half, `scripts/update_ruff_version.py`.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_REPO_CONFIG = _REPO / ".pre-commit-config.yaml"

# `spec_from_file_location`, the idiom nine test files in this repo already use for a `scripts/`
# module. A module-scope `sys.path.insert` has no precedent here and would reopen the path the gate
# closes with `PYTHONSAFEPATH=1` for every xdist worker.
_spec = importlib.util.spec_from_file_location("_urv", _REPO / "scripts" / "update_ruff_version.py")
_urv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_urv)

_CLEAN = """repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.16.0
    hooks:
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
"""


def _write(tmp_path, monkeypatch, text: str) -> Path:
    monkeypatch.chdir(tmp_path)
    p = tmp_path / ".pre-commit-config.yaml"
    p.write_text(text)
    return p


def _revs(text: str) -> list[str]:
    return re.findall(r"^\s*rev:\s*(v[0-9.]+)", text, flags=re.M)


def test_the_module_imports_without_requests():
    """The blocker this PR nearly shipped. `requests` is in neither pyproject.toml nor
    environment.yml (control: scipy is in both) and reaches this environment only through conda and
    Sphinx, so a module-scope `import requests` made every consumer of that file need an undeclared
    package. This test file was the first thing to need it, and collection failed.

    Reaching this line at all is the assertion: `_urv` is imported at module scope above.
    """
    assert callable(_urv.update_files)
    src = (_REPO / "scripts" / "update_ruff_version.py").read_text()
    assert not re.search(r"^(?:import requests\b|from requests\b)", src, flags=re.M), (
        "requests must stay inside the functions that use it until something declares it"
    )


def test_the_fixture_and_the_real_config_have_the_same_shape():
    """A control on the fixture, comparing it to the real file rather than only asserting about one.

    Both must have exactly one ruff repo and at least two `rev:` lines -- without a second one the
    defect cannot exist and every test below proves nothing.
    """
    real = _REPO_CONFIG.read_text()
    assert real.count("astral-sh/ruff-pre-commit") == _CLEAN.count("astral-sh/ruff-pre-commit") == 1
    assert len(_revs(real)) >= 2, "the defect needs a second `rev:` line to exist"
    assert len(_revs(_CLEAN)) >= 2, "so does the fixture"


def test_the_bump_leaves_every_other_rev_alone(tmp_path, monkeypatch):
    """The pin. An unanchored expression sets BOTH lines, which is the shipped defect."""
    p = _write(tmp_path, monkeypatch, _CLEAN)
    _urv.update_files("0.17.0")
    out = p.read_text()

    assert _revs(out) == ["v0.17.0", "v6.0.0"], (
        "pre-commit-hooks must not move -- an unanchored `s/rev: v[0-9.]+/` sets it to the ruff "
        "version, a tag that does not exist, and pre-commit then fetches no hook at all"
    )
    changed = [(a, b) for a, b in zip(_CLEAN.splitlines(), out.splitlines(), strict=True) if a != b]
    assert len(changed) == 1, f"expected exactly one changed line, got {changed}"


@pytest.mark.parametrize("version", ["0.17.0", "1.0.0", "0.16.1"])
def test_the_real_config_takes_the_bump_on_one_line(tmp_path, monkeypatch, version):
    """Against the repository's own file, so the fixture and reality cannot diverge silently."""
    before = _REPO_CONFIG.read_text()
    p = _write(tmp_path, monkeypatch, before)
    _urv.update_files(version)
    after = p.read_text()

    changed = [(a, b) for a, b in zip(before.splitlines(), after.splitlines(), strict=True) if a != b]
    assert len(changed) == 1, f"expected exactly one changed line, got {changed}"
    assert changed[0][1].strip() == f"rev: v{version}"
    lines = after.splitlines()
    assert "ruff-pre-commit" in lines[lines.index(changed[0][1]) - 1], "the changed rev must be ruff's"


def test_a_comment_between_repo_and_rev_does_not_silently_no_op(tmp_path, monkeypatch):
    """A shape the two bumpers used to disagree on, and the Python one failed SILENT.

    A comment between the `repo:` line and its `rev:` is valid YAML. `\\s+` cannot span it, so the
    substitution matched nothing, `update_files` returned `[]`, and `main()` printed "No files
    needed updating" and exited 0 -- while the workflow's `sed` handled the same file correctly.
    """
    fixture = _CLEAN.replace(
        "  - repo: https://github.com/astral-sh/ruff-pre-commit\n",
        "  - repo: https://github.com/astral-sh/ruff-pre-commit\n    # pinned monthly by check-ruff-updates.yml\n",
    )
    p = _write(tmp_path, monkeypatch, fixture)
    assert _urv.update_files("0.17.0") == [".pre-commit-config.yaml"]
    assert _revs(p.read_text()) == ["v0.17.0", "v6.0.0"]


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("no rev in the ruff block", _CLEAN.replace("    rev: v0.16.0\n", "", 1)),
        (
            "rev before repo",
            "repos:\n  - rev: v0.16.0\n    repo: https://github.com/astral-sh/ruff-pre-commit\n  - repo: https://github.com/pre-commit/pre-commit-hooks\n    rev: v6.0.0\n",
        ),
    ],
)
def test_a_bump_that_matches_nothing_raises_instead_of_reporting_success(name, text, tmp_path, monkeypatch):
    """A bump that matched nothing must raise, and the postcondition is what makes that sound.

    NOT because "main() only calls this when the versions differ" -- `--force` does not compare and
    calls straight through. The check is on the resulting VALUE of the pin, so `--force` to the
    current version writes nothing and passes, while a shape the pattern misses raises.

    Both shapes defeat a guard that merely counts changed lines -- the workflow's `sed` range lands
    on `pre-commit-hooks` in each and changes exactly one line -- which is why that guard reads the
    pin back instead of counting. This is the same invariant on the Python side.
    """
    _write(tmp_path, monkeypatch, text)
    with pytest.raises(RuntimeError, match="matched nothing"):
        _urv.update_files("0.17.0")


def test_the_bumper_no_longer_reaches_for_a_pin_that_moved(tmp_path, monkeypatch):
    """`modern_quality.yml` held a `ruff==` line that moved out of it, and `ci.yml` holds no pin --
    it reads the version out of `.pre-commit-config.yaml` at runtime. One live owner, one edit."""
    _write(tmp_path, monkeypatch, _CLEAN)
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    decoy = tmp_path / ".github" / "workflows" / "modern_quality.yml"
    decoy.write_text("jobs:\n  x:\n    steps:\n      - run: pip install ruff==0.16.0\n")

    assert _urv.update_files("0.17.0") == [".pre-commit-config.yaml"]
    assert "ruff==0.16.0" in decoy.read_text(), "a file that is not the owner must be left alone"


@pytest.mark.parametrize(
    ("name", "between"),
    [
        ("nothing", ""),
        ("a comment", "    # pinned monthly by check-ruff-updates.yml\n"),
        ("a blank line", "\n"),
        ("a blank line then a comment", "\n    # pinned monthly\n"),
        ("a comment then a blank line", "    # pinned monthly\n\n"),
    ],
)
def test_both_readers_in_the_file_agree_on_where_the_pin_is(name, between, tmp_path, monkeypatch):
    """One owner, pinned on the axis that broke it: WIDENING one reader narrowed it on the other.

    Three encodings of "where the ruff pin is" lived in this file and disagreed pairwise, all on
    valid YAML. `\\s+` spans a blank line and not a comment; the comment-tolerant form written for
    #2123 spanned a comment and not a blank line. So `get_current_version` printed `v0.16.0` for a
    blank-line config that `update_files` then refused, raising "the bump matched nothing. Check
    the shape of the ruff block" -- pointing the reader at a config that is fine.

    A single-shape test cannot catch that class: it pins one axis and the next widening is free to
    narrow the other. This asserts the two readers AGREE, which goes red whichever one moves.
    """
    fixture = _CLEAN.replace(
        "  - repo: https://github.com/astral-sh/ruff-pre-commit\n",
        "  - repo: https://github.com/astral-sh/ruff-pre-commit\n" + between,
    )
    p = _write(tmp_path, monkeypatch, fixture)
    assert _urv.get_current_version() == "0.16.0", f"the reader missed the pin with {name} between"
    assert _urv.update_files("0.17.0") == [".pre-commit-config.yaml"], (
        f"the bumper missed the pin with {name} between, and the reader did not"
    )
    assert _revs(p.read_text()) == ["v0.17.0", "v6.0.0"]


def test_a_bump_to_the_version_already_pinned_writes_nothing_and_does_not_raise(tmp_path, monkeypatch):
    """The postcondition checks the pin's resulting VALUE, not that a write happened -- and that
    distinction had no test, in a branch whose commit exists to make it.

    `--force` does not compare versions; it calls straight through. So `--force 0.16.0` against a
    config already at 0.16.0 legitimately writes nothing, and a postcondition phrased as "something
    must have been written" would fail a correct run. Every other call in this file bumps 0.16.0 to
    a different version, so all of them pass under either phrasing.
    """
    p = _write(tmp_path, monkeypatch, _CLEAN)
    assert _urv.update_files("0.16.0") == []
    assert p.read_text() == _CLEAN
