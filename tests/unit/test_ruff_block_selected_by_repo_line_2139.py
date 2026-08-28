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


# The shape table. Anchoring the pin expression to the ruff `repo:` LINE is a widening on the axis
# #2123 is about and a NARROWING on every axis the anchor adds -- and the first version of this fix
# took four shapes away from the `grep -A1` expressions it replaced. Each is ordinary YAML that
# `pre-commit` accepts. A widening is not a superset: the axis the old matcher spanned has to be
# named and checked, not assumed.
HEADER = """\
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace

"""
TAIL = "    hooks:\n      - id: ruff\n"

SHAPES = {
    "plain": "  - repo: https://github.com/astral-sh/ruff-pre-commit\n    rev: v0.16.0\n",
    "comment between repo and rev (#2123)": (
        "  - repo: https://github.com/astral-sh/ruff-pre-commit\n    # kept in step with uv.lock\n    rev: v0.16.0\n"
    ),
    "quoted url": '  - repo: "https://github.com/astral-sh/ruff-pre-commit"\n    rev: v0.16.0\n',
    "dot-git suffix": ("  - repo: https://github.com/astral-sh/ruff-pre-commit.git\n    rev: v0.16.0\n"),
    "trailing comment on the repo line": (
        "  - repo: https://github.com/astral-sh/ruff-pre-commit  # the linter\n    rev: v0.16.0\n"
    ),
    "http not https": "  - repo: http://github.com/astral-sh/ruff-pre-commit\n    rev: v0.16.0\n",
    "quoted rev": '  - repo: https://github.com/astral-sh/ruff-pre-commit\n    rev: "v0.16.0"\n',
    "trailing space after the url": ("  - repo: https://github.com/astral-sh/ruff-pre-commit   \n    rev: v0.16.0\n"),
}

REFUSED = {
    "two components": "  - repo: https://github.com/astral-sh/ruff-pre-commit\n    rev: v0.16\n",
    "a bare dot": "  - repo: https://github.com/astral-sh/ruff-pre-commit\n    rev: v.\n",
    "a sha": "  - repo: https://github.com/astral-sh/ruff-pre-commit\n    rev: 3f8b2c1\n",
    "no rev at all": "  - repo: https://github.com/astral-sh/ruff-pre-commit\n    hooks:\n",
}


@pytest.mark.parametrize("shape", SHAPES)
def test_the_reader_accepts_every_shape_the_greps_accepted(tmp_path, shape):
    (tmp_path / ".pre-commit-config.yaml").write_text(HEADER + SHAPES[shape] + TAIL)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--print-current"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, f"{shape}: {out.stderr}"
    assert out.stdout.strip() == "0.16.0", shape


@pytest.mark.parametrize("shape", SHAPES)
def test_a_bump_moves_only_the_ruff_pin_in_every_shape(tmp_path, shape):
    config = tmp_path / ".pre-commit-config.yaml"
    config.write_text(HEADER + SHAPES[shape] + TAIL)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--force", "0.17.0"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, f"{shape}: {out.stdout}{out.stderr}"
    revs = [line.strip() for line in config.read_text().splitlines() if line.strip().startswith("rev:")]
    assert revs[0] == "rev: v6.0.0", f"{shape}: the bump moved pre-commit-hooks"
    assert "0.17.0" in revs[1], f"{shape}: the ruff pin did not move -- {revs}"


@pytest.mark.parametrize("shape", REFUSED)
def test_a_rev_that_is_not_a_version_is_refused_not_printed(tmp_path, shape):
    """The read side lost this check when six expressions became one, and nothing noticed.

    `ci.yml`'s deleted expression required `v[0-9]+\\.[0-9]+\\.[0-9]+`; the consolidated reader
    accepted anything of digits and dots, so `rev: v.` printed `.` and `rev: v0.16` printed `0.16`
    into a `pip install ruff==` whose only guard is `-z`. That is #2134's shape reached from the
    read side, in the commit that removed it from the fetch side.
    """
    (tmp_path / ".pre-commit-config.yaml").write_text(HEADER + REFUSED[shape] + TAIL)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--print-current"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0, f"{shape}: printed {out.stdout.strip()!r} instead of refusing"
    assert out.stdout.strip() == "", f"{shape}: stdout must stay empty"
