"""#2123: the ruff bump must touch the ruff pin and nothing else.

`.pre-commit-config.yaml` carries two `rev:` lines -- `astral-sh/ruff-pre-commit` and
`pre-commit/pre-commit-hooks`. The monthly workflow's `sed` was unanchored, and `s///` applies once
PER LINE, so a bump set `pre-commit-hooks` to the ruff version too: a tag that does not exist, so
`pre-commit` could fetch no hook environment and the bot's own PR opened with every hook broken.

The workflow half cannot be unit-tested from here -- it is a `sed` in a GitHub Actions step -- and
carries its own guard instead: it aborts if the bump changes more than one line. This file pins the
half that IS reachable, `scripts/update_ruff_version.py`, whose regex was already anchored and had
nothing asserting it.

The fixture is checked against the real file, so it cannot drift into testing a shape the repository
no longer has.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from update_ruff_version import update_files

_REPO_CONFIG = Path(__file__).resolve().parents[2] / ".pre-commit-config.yaml"

_FIXTURE = """repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.16.0
    hooks:
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
"""


def test_the_fixture_still_matches_the_shape_of_the_real_config():
    """The control on the fixture. Two repos, two `rev:` lines -- if that stops being true the
    test below is exercising a shape this repository does not have."""
    real = _REPO_CONFIG.read_text()
    assert real.count("astral-sh/ruff-pre-commit") == 1
    assert len(re.findall(r"^\s*rev:\s*v[0-9.]+", real, flags=re.M)) >= 2, (
        "the defect needs a second `rev:` line to exist; without one this test proves nothing"
    )


def test_the_bump_leaves_every_other_rev_alone(tmp_path, monkeypatch):
    """The pin. An unanchored expression sets BOTH lines, which is the shipped defect."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".pre-commit-config.yaml").write_text(_FIXTURE)

    update_files("0.17.0")
    out = (tmp_path / ".pre-commit-config.yaml").read_text()

    assert "rev: v0.17.0" in out, "the ruff pin must move"
    assert "rev: v6.0.0" in out, (
        "pre-commit-hooks must not move -- an unanchored `s/rev: v[0-9.]+/` sets it to the ruff "
        "version, a tag that does not exist, and pre-commit then fetches nothing"
    )
    changed = [(a, b) for a, b in zip(_FIXTURE.splitlines(), out.splitlines(), strict=True) if a != b]
    assert len(changed) == 1, f"expected exactly one changed line, got {changed}"


def test_the_unanchored_form_is_what_this_rejects():
    """Names the defect rather than only the fix, so the assertion above has a stated counterexample.

    Not a test of production code -- it applies the retired expression to the fixture and shows it
    hits both lines. If this ever stops hitting two, the defect has changed shape and the pin above
    needs re-deriving rather than trusting.
    """
    unanchored = [
        (a, re.sub(r"rev: v[0-9.]+", "rev: v0.17.0", a, count=1))
        for a in _FIXTURE.splitlines()
        if re.sub(r"rev: v[0-9.]+", "rev: v0.17.0", a, count=1) != a
    ]
    assert len(unanchored) == 2, f"the retired expression should hit both `rev:` lines, hit {unanchored}"


def test_the_bumper_no_longer_reaches_for_a_pin_that_moved(tmp_path, monkeypatch):
    """#2123's other half: `modern_quality.yml` held a `ruff==` line that moved out of it.

    `ci.yml` holds no pin either -- it reads the version out of `.pre-commit-config.yaml` at
    runtime -- so there is exactly one owner and the bumper must report exactly one file.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".pre-commit-config.yaml").write_text(_FIXTURE)
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    decoy = tmp_path / ".github" / "workflows" / "modern_quality.yml"
    decoy.write_text("jobs:\n  x:\n    steps:\n      - run: pip install ruff==0.16.0\n")

    written = update_files("0.17.0")

    assert written == [".pre-commit-config.yaml"], f"the bumper must touch one owner, touched {written}"
    assert "ruff==0.16.0" in decoy.read_text(), "a file that is not the owner must be left alone"


@pytest.mark.parametrize("version", ["0.17.0", "1.0.0", "0.16.1"])
def test_the_real_config_takes_the_bump_on_one_line(tmp_path, monkeypatch, version):
    """Against the repository's own file rather than the fixture, so the two cannot diverge."""
    monkeypatch.chdir(tmp_path)
    before = _REPO_CONFIG.read_text()
    (tmp_path / ".pre-commit-config.yaml").write_text(before)

    update_files(version)
    after = (tmp_path / ".pre-commit-config.yaml").read_text()

    changed = [(a, b) for a, b in zip(before.splitlines(), after.splitlines(), strict=True) if a != b]
    assert len(changed) == 1, f"expected exactly one changed line, got {changed}"
    assert changed[0][1].strip() == f"rev: v{version}"
    # and it is the ruff one: the line above the changed line names the repo
    lines = after.splitlines()
    idx = lines.index(changed[0][1])
    assert "ruff-pre-commit" in lines[idx - 1], f"the changed rev belongs to {lines[idx - 1].strip()}"
