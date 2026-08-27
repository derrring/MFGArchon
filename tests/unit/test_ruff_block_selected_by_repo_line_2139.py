"""#2139: the ruff pin was located by a bare occurrence of the repo URL, not by its `repo:` line.

`.pre-commit-config.yaml` pins more than one repository. A comment inside an earlier block that
merely names the ruff URL -- `# kept in step with astral-sh/ruff-pre-commit` -- satisfied that bare
form, and `re.sub` replaces every match, so a bump rewrote the earlier block's `rev:` as well.
`pre-commit-hooks` went from a real tag to the ruff version, which does not exist as a
pre-commit-hooks tag, and `pre-commit` could then fetch no hook environment at all. That is #2123's
damage arriving from the reader's side.

What made it silent is worth stating separately: the post-bump check inspected the same wrong
block, and the corruption had just written the asked-for version into it, so the check passed. A
first attempt at this issue anchored only that check -- which made it inspect the correct block,
find the correct value, and still pass while the corruption happened elsewhere. Both assertions
below fail against that attempt, which is why they test the config rather than the selector.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "update_ruff_version.py"

# The decoy is a comment, in an earlier block, naming the ruff repo. Nothing about it is exotic:
# keeping two pins in step is exactly the sort of thing a maintainer writes down.
DECOY = """\
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    # kept in step with astral-sh/ruff-pre-commit
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace

  - repo: https://github.com/astral-sh/ruff-pre-commit
    # a comment here too: the shape #2123 was about, which the pin must still span
    rev: v0.16.0
    hooks:
      - id: ruff
"""


@pytest.fixture
def decoy_repo(tmp_path):
    (tmp_path / ".pre-commit-config.yaml").write_text(DECOY)
    return tmp_path


def test_the_reader_is_not_fooled_by_a_comment_naming_the_repo(decoy_repo):
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--print-current"],
        cwd=decoy_repo,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0.16.0", (
        f"read {out.stdout.strip()!r} -- the decoy comment routed the pin to the "
        "pre-commit-hooks block, whose rev is v6.0.0"
    )


def test_a_bump_does_not_move_the_other_repo(decoy_repo):
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--force", "0.17.0"],
        cwd=decoy_repo,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr

    written = (decoy_repo / ".pre-commit-config.yaml").read_text()
    revs = [line.strip() for line in written.splitlines() if line.strip().startswith("rev:")]
    assert revs == ["rev: v6.0.0", "rev: v0.17.0"], (
        f"expected only the ruff pin to move, got {revs} -- v0.17.0 is not a pre-commit-hooks tag, "
        "so `pre-commit` would fetch no hook environment at all"
    )
