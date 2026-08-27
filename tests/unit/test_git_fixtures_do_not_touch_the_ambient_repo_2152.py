"""#2152: git fixtures must not write into the repository `GIT_DIR` names.

`git -C <tmp>` sets the working directory. It does not override `GIT_DIR`, which names the
repository directly and wins. Git exports `GIT_DIR` to every hook it runs, so a suite invoked from
the pre-push hook has it set, pointing at the checkout being pushed -- and every fixture that looks
isolated because it passes `-C <tmp>` writes there instead.

#2085 fixed this in `test_prune_local_branches.py`, where it was found. It was not fixed in
`scripts/check_citations.py` or `tests/unit/test_check_citations.py`, which between them run
`init`, `add`, `commit`, `checkout`, `merge` and `update-index`. Observed 2026-08-27 on a real
push: four commits authored `t <t@t>` -- the fixture's own `user.email` -- landed on the branch
being pushed, a `base`/`ours`/`theirs` merge and an `other` branch appeared, every tracked file was
staged as deleted, and three fixture directories were left in the tree.

The assertions below are on the AMBIENT repository, never on the temporary one. Asserting the
fixture worked is what passing looks like when the isolation is gone: with `GIT_DIR` set, the read
comes back from the same wrong place the write went to.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# The variables that redirect git away from `-C` and `cwd=`. Named here rather than imported so
# this test keeps failing if the shared list is deleted or renamed.
_LEAKS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_CONFIG")


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _LEAKS}


@pytest.fixture
def ambient(tmp_path):
    """A repository standing in for the developer's checkout, with one commit and nothing else."""
    root = tmp_path / "ambient"
    root.mkdir()
    git = ["git", "-C", str(root), "-c", "user.email=owner@example.invalid", "-c", "user.name=owner"]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True, env=_clean_env())
    (root / "kept.txt").write_text("do not touch\n")
    subprocess.run([*git, "add", "-A"], check=True, env=_clean_env())
    subprocess.run([*git, "commit", "-qm", "the developer's work"], check=True, env=_clean_env())
    return root


def _state(root: Path) -> dict:
    """Everything a leaked fixture would disturb: history, refs, the index, and the tree."""

    def out(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, env=_clean_env()).stdout

    return {
        "log": out("log", "--format=%H %an <%ae> %s"),
        "branches": out("branch", "--list"),
        "status": out("status", "--porcelain"),
        "files": sorted(p.name for p in root.iterdir() if p.name != ".git"),
    }


def _run_under_leaked_git_dir(ambient: Path, argv: list[str], cwd: Path):
    env = {**_clean_env(), "GIT_DIR": str(ambient / ".git")}
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, env=env)


def test_the_citation_self_test_leaves_the_ambient_repo_alone(ambient, tmp_path):
    """`scripts/check_citations.py --self-test` runs `git init` and `git add -A` in a temp tree."""
    before = _state(ambient)
    done = _run_under_leaked_git_dir(
        ambient, [sys.executable, str(REPO / "scripts" / "check_citations.py"), "--self-test"], tmp_path
    )
    assert done.returncode == 0, f"the self-test itself failed:\n{done.stdout}\n{done.stderr}"
    assert _state(ambient) == before, (
        "`check_citations.py --self-test` wrote into the repository GIT_DIR names. Run from the "
        "pre-push hook that is the developer's checkout. (#2152)"
    )


def test_the_citation_tests_leave_the_ambient_repo_alone(ambient, tmp_path):
    """The heaviest fixture in the suite: `init`, `add`, `commit`, `checkout -b`, `merge`."""
    before = _state(ambient)
    done = _run_under_leaked_git_dir(
        ambient,
        [
            sys.executable,
            "-m",
            "pytest",
            str(REPO / "tests" / "unit" / "test_check_citations.py"),
            "-q",
            "-p",
            "no:cacheprovider",
            "-k",
            "mid_conflict or bare_basename or changelog_is_exempt",
        ],
        REPO,
    )
    assert done.returncode == 0, f"the citation tests themselves failed:\n{done.stdout[-3000:]}"
    after = _state(ambient)
    assert after == before, (
        "`test_check_citations.py` wrote into the repository GIT_DIR names.\n"
        f"log before:   {before['log']!r}\nlog after:    {after['log']!r}\n"
        f"branches:     {before['branches']!r} -> {after['branches']!r}\n"
        f"status lines: {len(before['status'].splitlines())} -> {len(after['status'].splitlines())}\n"
        f"tree:         {before['files']} -> {after['files']}  (#2152)"
    )


def test_the_script_still_imports_under_the_gates_own_interpreter_flags(tmp_path):
    """The gate exports `PYTHONSAFEPATH=1`, and every subprocess inherits it.

    Under it the interpreter does not prepend a script's own directory to `sys.path`, so a sibling
    `import` inside `scripts/` resolves when the file is run by hand and raises
    `ModuleNotFoundError` under the gate. That asymmetry is why this is a test and not a review
    note: fifteen tests passed in isolation and the same fifteen were red in the full suite, and
    nothing about running them locally could have shown it.
    """
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_citations.py"), "--self-test"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**_clean_env(), "PYTHONSAFEPATH": "1"},
    )
    assert done.returncode == 0, "the script does not run under the flags the gate uses:\n" + done.stdout + done.stderr
