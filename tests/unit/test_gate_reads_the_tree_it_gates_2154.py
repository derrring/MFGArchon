"""#2154: the gate must import the tree it is gating, say which one, and refuse if it is not.

`"$PY" scripts/X.py` puts `scripts/` on `sys.path[0]`, not the repository root. `scripts/` holds no
`mfgarchon`, so the import falls through `PathFinder` to setuptools' editable finder, whose mapping
is hard-wired to the original checkout. From a `git worktree` the capability matrix and the
deprecation self-test therefore measured a different tree, on whatever branch it was sitting on.

**The second tree is two files in `tmp_path`, not a `git worktree`.** The defect is "gate root is not
the editable-install root"; a plain directory holding `scripts/local_ci.sh` reproduces it, and the
whole file runs in about a second. Building it with `git worktree add` would be ~50x the cost and
would leave debris on failure.

Two stoppers keep it cheap, both borrowed from the gate rather than stubbed: `MFG_PYTHON` skips the
interpreter search, whose probe otherwise imports the real package; and `# MUTATED` in the fake
package trips the mutated-tree guard, so a passing run ends just past the line under test.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "scripts" / "local_ci.sh"

GATE_LINE = re.compile(r"^printf '(?:\\n)?gate (\w+)\s*:", re.M)
PACKAGE_LINE = re.compile(r"^gate package\s*:\s*(\S.*?)\s*$", re.M)
RUFF_LINE = re.compile(r"^gate ruff\s*:\s*(\S.*?)\s*$", re.M)
REFUSAL = "the gate imported mfgarchon from"


def _tree(tmp_path: Path, name: str, *, package: bool) -> Path:
    root = tmp_path / name
    (root / "scripts").mkdir(parents=True)
    shutil.copy(GATE, root / "scripts" / "local_ci.sh")
    if package:
        (root / "mfgarchon").mkdir()
        (root / "mfgarchon" / "__init__.py").write_text("# MUTATED -- stopper, see this module's docstring\n")
    return root


def _run(root: Path, *, cwd: Path | None = None, pythonpath: str | None = None):
    env = {**os.environ, "MFG_PYTHON": sys.executable}
    env.pop("PYTHONPATH", None)
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    done = subprocess.run(
        ["bash", str(root / "scripts" / "local_ci.sh"), "--fast"],
        cwd=str(cwd or root),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return done.returncode, done.stdout + done.stderr


def _package_line(out: str) -> str:
    """A missing line has two causes and merging them would hide a deleted printf.

    No `gate interpreter` either means the gate never reached its head -- an environment failure,
    which exits 2 exactly as a refusal does. `gate interpreter` present means the head ran and the
    line was removed, which is the defect.
    """
    m = PACKAGE_LINE.search(out)
    if m:
        return m.group(1)
    if "gate interpreter" in out:
        raise AssertionError(f"the head ran but printed no `gate package` line (#2154):\n{out[:800]}")
    pytest.skip(f"the gate never reached its own head under {sys.executable}:\n{out[:600]}")


def test_the_gate_imports_the_tree_it_is_gating(tmp_path):
    root = _tree(tmp_path, "second_tree", package=True)
    _, out = _run(root)
    assert _package_line(out) == str((root / "mfgarchon").resolve()), (
        f"the gate imported a package from outside the tree it was gating ({root}):\n{out[:800]}"
    )
    assert REFUSAL not in out, f"refused a tree whose own package it should have found:\n{out[:800]}"


def test_the_gate_refuses_a_package_from_outside_the_tree(tmp_path):
    """Exit 2, not 1: nothing was measured, which is what `cannot_run` means and what
    `scripts/gate_hook.sh` branches on. Exit 1 is the content verdict, `GATE RED -- do not push`,
    and it would tell the operator their code is bad."""
    root = _tree(tmp_path, "tree_without_a_package", package=False)
    decoy = tmp_path / "decoy"
    (decoy / "mfgarchon").mkdir(parents=True)
    (decoy / "mfgarchon" / "__init__.py").write_text("")

    rc, out = _run(root, pythonpath=str(decoy))

    assert _package_line(out) == str((decoy / "mfgarchon").resolve()), (
        "the decoy is not being reached; this test would prove nothing"
    )
    assert rc == 2, f"a gate that imported another tree must exit 2, got {rc}:\n{out[:800]}"
    assert REFUSAL in out, f"the refusal must say what happened:\n{out[:800]}"
    assert str(decoy) in out, "the refusal must name the tree it imported"
    assert str(root) in out, "the refusal must name the tree it was gating, or recovery is a search"
    assert "PASS" not in out, f"a check ran after the refusal:\n{out[:800]}"


def test_entering_through_a_symlink_is_not_a_refusal(tmp_path):
    """The permission half. Without it a guard that refused everything would pass the other tests.

    The probe reports a `resolve()`d path and `$PWD` is the logical one, so comparing them
    unresolved fails on a correct tree reached through any symlinked parent.
    """
    root = _tree(tmp_path, "real_tree", package=True)
    link = tmp_path / "via_symlink"
    link.symlink_to(root, target_is_directory=True)
    _, out = _run(link, cwd=link)
    assert _package_line(out) == str((root / "mfgarchon").resolve())
    assert REFUSAL not in out, f"false refusal on a tree entered through a symlink:\n{out[:800]}"


def test_the_tool_invocations_are_not_shadowable_from_the_tree(tmp_path):
    """The regression this change was reworked to avoid, and the reason the path is an array.

    `local_ci.sh` invokes ruff, mypy and pytest as `-P -m`, and its own comment says `-P` is
    load-bearing: `-m` puts CWD at the front of `sys.path`, CWD is the repository root, so a `ruff/`
    package committed there shadows the linter and the gate reports GREEN over unlinted files while
    `gate ruff : ruff 0.16.0` sits in the pasted tail as forged evidence.

    `-P` removes CWD. It does **not** remove `PYTHONPATH`. An exported `PYTHONPATH=$PWD` therefore
    puts the root back and disarms that guard -- measured, a planted package answered `-P -m ruff`,
    and `PYTHONSAFEPATH=1 -P -m pytest` too, which is the variant added *because* `-P` does not reach
    xdist workers. So the tree is bound to the `scripts/*.py` invocations and to nothing else.
    """
    root = _tree(tmp_path, "tree_with_a_planted_linter", package=True)
    planted = root / "ruff"
    planted.mkdir()
    (planted / "__init__.py").write_text("")
    (planted / "__main__.py").write_text("print('SHADOWED-RUFF 99.99.99')\n")

    _, out = _run(root)
    m = RUFF_LINE.search(out)
    assert m, f"no `gate ruff` line to check:\n{out[:800]}"
    assert "SHADOWED" not in m.group(1), (
        f"a package planted at the tree root answered the linter invocation: {m.group(1)!r}. "
        "The tree is reaching the `-P -m` tools, which is what `-P` exists to prevent."
    )


def test_every_gate_line_reaches_the_pasted_tail():
    """Counted, not matched against literals: a label added to the head only, or dropped from the
    tail only, lands at one occurrence and fires here. Nothing checked any of these before -- not
    even `gate interpreter`, whose own comment calls it the only tell for a forged interpreter.
    """
    labels = GATE_LINE.findall(GATE.read_text())
    assert labels, "no `gate <label> :` lines found -- the pattern is wrong, not the gate"
    counts = {label: labels.count(label) for label in sorted(set(labels))}
    not_twice = {label: n for label, n in counts.items() if n != 2}
    assert not not_twice, (
        "every `gate <label>` line must print twice -- once at the head for a live run and once in "
        f"the tail a reviewer pastes. These do not: {not_twice} (all: {counts})"
    )
    assert "package" in counts, "the tree the gate imported must be in the pasted evidence (#2154)"
