"""The ruff pin has one reader, and it survives a comment between `repo:` and `rev:`.

`.pre-commit-config.yaml` holds the ruff version, and it was read from six places in five
different expressions. Four of them were `grep -A1`, which sees only the line immediately after
the repo URL -- so a comment there, which the *writer* (`RUFF_PIN` in `update_ruff_version.py`)
deliberately tolerates, silently breaks four of the five readers.

That asymmetry is #2135. The writer accepts a shape the verifiers reject, so a config the bumper
produces can be one the gate and the CI cannot read, and the failure is a version that reads as
empty rather than an error.

These two tests are the acceptance criterion for the consolidation:

  - `test_every_reader_survives_a_comment_in_the_block` exercises the behaviour;
  - `test_no_hand_rolled_reader_survives` counts, because behaviour passing while a fifth
    expression sits in a workflow nobody runs locally is how this came back.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "update_ruff_version.py"

_WITH_COMMENT = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
  - repo: https://github.com/astral-sh/ruff-pre-commit
    # bumped monthly by check-ruff-updates.yml
    rev: v0.16.0
    hooks:
      - id: ruff
"""


def test_every_reader_survives_a_comment_in_the_block(tmp_path):
    """One reader, and a comment between `repo:` and `rev:` does not hide the pin from it."""
    cfg = tmp_path / ".pre-commit-config.yaml"
    cfg.write_text(_WITH_COMMENT)

    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--print-current"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.strip() == "0.16.0", f"the pin reads {out.stdout.strip()!r} through the owner; the comment hid it"

    # Control: the same reader on the same config without the comment. If this fails the
    # fixture is wrong, not the reader.
    cfg.write_text(_WITH_COMMENT.replace("    # bumped monthly by check-ruff-updates.yml\n", ""))
    clean = subprocess.run(
        [sys.executable, str(SCRIPT), "--print-current"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert clean.stdout.strip() == "0.16.0", clean.stdout + clean.stderr


def test_no_hand_rolled_reader_survives():
    """Nothing outside the owner reads the pin with its own expression."""
    hand_rolled = []
    for path in [
        REPO / "scripts" / "local_ci.sh",
        REPO / ".github" / "workflows" / "ci.yml",
        REPO / ".github" / "workflows" / "check-ruff-updates.yml",
    ]:
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"grep\s+-A\s?1.*ruff-pre-commit", line):
                hand_rolled.append(f"{path.relative_to(REPO)}:{n}")

    assert not hand_rolled, (
        "these read the ruff pin with their own expression instead of asking the owner "
        f"(`{SCRIPT.relative_to(REPO)} --print-current`): " + ", ".join(hand_rolled)
    )


# The pin's WRITE side, same argument as the read side above. `.github/workflows/` carried its own
# `sed -i` bumper, so one quantity had two writers; they disagreed on whether a comment may sit
# between `repo:` and `rev:`, and #2123 had to be fixed once on each. A second writer is invisible
# until the two disagree, which is why this is a test and not a review note.
WRITES_THE_CONFIG = re.compile(r"sed\s+-i.*pre-commit-config", re.I)


def test_the_config_has_one_writer():
    hand_rolled = []
    for path in [*sorted((REPO / ".github" / "workflows").glob("*.yml")), REPO / "scripts" / "local_ci.sh"]:
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if WRITES_THE_CONFIG.search(line):
                hand_rolled.append(f"{path.relative_to(REPO)}:{n}")
    assert not hand_rolled, (
        "these write .pre-commit-config.yaml directly instead of calling the owner "
        f"(`{SCRIPT.relative_to(REPO)} --force VERSION`): " + ", ".join(hand_rolled)
    )


def test_a_failure_leaves_stdout_empty(tmp_path):
    """stdout is the data channel; a diagnostic on it is captured as the value.

    Four call sites now parse this command, and three of them read it inside `$(...)`. A shell
    reading `RUFF_VERSION=$(... ) || RUFF_VERSION=""` gets the *stdout* of a failing run, so an
    error message printed there arrives as a non-empty string, the `-z` guard downstream stays
    quiet, and the text flows on as a version. That is #2134's shape reached by a different route:
    a value nobody validated because the thing that produced it looked like it had succeeded.
    """
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--print-current"],
        cwd=tmp_path,  # no .pre-commit-config.yaml here
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0, "a missing config must fail, not return something usable"
    assert out.stdout.strip() == "", (
        f"stdout must stay empty on failure, got {out.stdout.strip()!r} -- a caller reading "
        "`$(...)` would take that as the version"
    )
    assert "pre-commit-config" in out.stderr, "the diagnostic must still reach stderr"
