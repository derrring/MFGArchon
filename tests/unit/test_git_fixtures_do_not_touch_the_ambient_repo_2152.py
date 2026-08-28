"""#2152: git fixtures must not reach the repository the environment names.

`git -C <dir>` and `cwd=` set the working directory. Neither overrides `GIT_DIR`, which names the
repository and wins. **A `pre-push` hook run from a linked worktree has it set**, pointing at the
checkout being pushed — measured on git 2.50.1; a push from the main checkout, or from a
subdirectory of it, exports no `GIT_DIR` at all. The lanes under `.claude/worktrees/` are linked
worktrees, which is why this fires for them and for nothing else, and why reproducing it from the
main checkout finds nothing.

Observed on a real push, 2026-08-28 local: four commits authored `t <t@t>` — the fixture's own
`user.email` — on the branch being pushed, a `base`/`ours`/`theirs` merge, a stray `other` branch,
every tracked file staged as deleted, three fixture directories in the tree. Recovered from the
worktree's own HEAD reflog.

Every assertion here is on the **ambient** repository, never on the temporary one. Asserting the
fixture worked is what passing looks like when the isolation is gone: with `GIT_DIR` set, the read
comes back from the same wrong place the write went to.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CITATIONS = REPO / "scripts" / "check_citations.py"

# The owner, loaded the way every other consumer loads it. Not re-listed here: an earlier version
# kept a four-name copy "so this test keeps failing if the shared list is deleted", which is not
# what happens — deleting the file is caught by `tests/conftest.py`'s import, as an error over the
# whole suite. The copy bought nothing and was the same private-copy shape #2085 was faulted for.
_SPEC = __import__("importlib.util", fromlist=["util"]).spec_from_file_location(
    "_mfgarchon_git_env_2152", REPO / "scripts" / "git_env.py"
)
GIT_ENV = __import__("importlib.util", fromlist=["util"]).module_from_spec(_SPEC)
_SPEC.loader.exec_module(GIT_ENV)


@pytest.fixture
def ambient(tmp_path):
    """A repository standing in for the developer's checkout: one commit, nothing else.

    Built under `isolated_env()`, not a hand-rolled partial scrub. Three ordinary global settings
    turn this fixture into a setup *error* rather than a test failure: `commit.gpgsign` with a key
    this process cannot use, and `init.templateDir` or `core.hooksPath` naming hooks that then run
    during the fixture's own commit. All three are what `isolated_env` exists for.
    """
    root = tmp_path / "ambient"
    root.mkdir()
    git = ["git", "-C", str(root), "-c", "user.email=owner@example.invalid", "-c", "user.name=owner"]
    env = GIT_ENV.isolated_env()
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True, env=env)
    (root / "kept.txt").write_text("do not touch\n")
    subprocess.run([*git, "add", "-A"], check=True, env=env)
    subprocess.run([*git, "commit", "-qm", "the developer's work"], check=True, env=env)
    return root


def _state(root: Path) -> dict:
    """What a leaked fixture disturbs. Not exhaustive — see the assertions below for what it misses.

    `check=True`, deliberately: a git call that starts failing would return `""` for both the before
    and the after, and the key would stop discriminating with nothing to say so.

    `files` used to be here and was dead. With `GIT_DIR` set and no `GIT_WORK_TREE`, git treats
    **cwd** as the worktree — the temporary directory, never this one — so it could not move under
    any leak this test can produce.
    """

    def out(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
            env=GIT_ENV.isolated_env(),
        ).stdout

    return {
        "log": out("log", "--format=%H %an <%ae> %s"),
        # Every ref, not just branches: tags, notes and `refs/stash` are writes a branch listing
        # cannot see.
        "refs": out("for-each-ref", "--format=%(refname) %(objectname)"),
        "status": out("status", "--porcelain"),
        # #2085's second symptom was a write to a config file, which nothing above would notice.
        "config": (root / ".git" / "config").read_text(),
        # Arbitrary code execution if a fixture plants one.
        "hooks": sorted(p.name for p in (root / ".git" / "hooks").iterdir() if not p.name.endswith(".sample")),
    }


def _under_leaked_git_dir(ambient: Path, argv: list[str], cwd: Path, extra: dict | None = None):
    env = {**GIT_ENV.without_git_leaks(), "GIT_DIR": str(ambient / ".git"), **(extra or {})}
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, env=env)


def test_the_citation_self_test_leaves_the_ambient_repo_alone(ambient, tmp_path):
    """`check_citations.py --self-test` runs `git init` and `git add -A` in a temporary tree."""
    before = _state(ambient)
    done = _under_leaked_git_dir(ambient, [sys.executable, str(CITATIONS), "--self-test"], tmp_path)
    assert done.returncode == 0, f"the self-test itself failed:\n{done.stdout}\n{done.stderr}"
    assert _state(ambient) == before, (
        "`check_citations.py --self-test` wrote into the repository GIT_DIR names. (#2152)"
    )


def test_the_conflict_fixture_leaves_the_ambient_repo_alone(ambient, tmp_path):
    """The fixture that actually fired: `init`, `add`, `commit`, `checkout -b`, `merge`.

    Selected by node id rather than `-k`. A `-k` expression that stops matching runs nothing and
    reports success on an empty selection; a node id that stops resolving makes pytest exit 4 and
    say so. The session-scoped scrub cannot be exercised from inside this process — `monkeypatch`
    runs after it — so a real child process is required, and this is the cheapest one that reaches
    the fixture.
    """
    node = "tests/unit/test_check_citations.py::test_a_tree_mid_conflict_is_refused_not_counted_three_times"
    before = _state(ambient)
    done = _under_leaked_git_dir(
        ambient,
        [sys.executable, "-m", "pytest", node, "-q", "-p", "no:cacheprovider"],
        REPO,
    )
    assert done.returncode == 0, f"the nested run failed ({done.returncode}):\n{done.stdout[-2500:]}"
    assert "1 passed" in done.stdout, f"the node id selected nothing:\n{done.stdout[-2000:]}"
    after = _state(ambient)
    assert after == before, (
        "the conflict fixture wrote into the repository GIT_DIR names.\n"
        f"log:    {before['log']!r} -> {after['log']!r}\n"
        f"refs:   {before['refs']!r} -> {after['refs']!r}\n"
        f"status: {len(before['status'].splitlines())} -> {len(after['status'].splitlines())} lines  (#2152)"
    )


def test_a_planted_hook_reached_through_the_config_family_does_not_run(tmp_path):
    """The escape an eleven-name list could not stop, and the reason this is a denylist.

    `GIT_CONFIG_COUNT` / `GIT_CONFIG_KEY_n` / `GIT_CONFIG_VALUE_n` carry `-c key=value` across a
    process boundary, and git itself sets `GIT_CONFIG_PARAMETERS` for any `git -c … push`, which
    reaches the hook. Through it, `init.templateDir` plants hooks that `git init` copies into every
    repository it creates and that run during that repository's own commit — the hole
    `GIT_TEMPLATE_DIR` was added to close, reached by the config spelling of the same setting.

    The probe commits, because that is what fires a `post-commit`. An earlier version of this test
    pointed at `check_citations.py --self-test`, which only does `init` and `add`: it passed with
    the config family allowlisted, and the mutation that should have killed it did not.
    """
    planted = tmp_path / "template" / "hooks"
    planted.mkdir(parents=True)
    marker = tmp_path / "EVIL_RAN"
    hook = planted / "post-commit"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)

    # A fixture in miniature: scrub the process the way every consumer does, then build a throwaway
    # repository with `git -C` exactly as the real ones do.
    work = tmp_path / "fixture"
    work.mkdir()
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import importlib.util, pathlib, subprocess\n"
        f"s = importlib.util.spec_from_file_location('ge', {str(REPO / 'scripts' / 'git_env.py')!r})\n"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        "m.scrub_process_env()\n"
        f"d = {str(work)!r}\n"
        "g = ['git', '-C', d, '-c', 'user.email=f@f', '-c', 'user.name=f']\n"
        "subprocess.run(g + ['init', '-q'], check=True)\n"
        "pathlib.Path(d, 'a.txt').write_text('a')\n"
        "subprocess.run(g + ['add', '-A'], check=True)\n"
        "subprocess.run(g + ['commit', '-qm', 'x'], check=True)\n"
    )
    done = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True,
        text=True,
        env={
            **GIT_ENV.without_git_leaks(),
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "init.templateDir",
            "GIT_CONFIG_VALUE_0": str(planted.parent),
        },
    )
    assert done.returncode == 0, f"the probe itself failed:\n{done.stdout}\n{done.stderr}"
    assert not marker.exists(), (
        "a hook planted through GIT_CONFIG_* was copied into a fixture repository by `git init` "
        "and executed during its commit — the scrub let the config family through"
    )


def test_the_scrub_covers_every_variable_git_itself_calls_local(tmp_path):
    """The population is git's, not ours. `git rev-parse --local-env-vars` is the list `githooks(5)`
    names, and it grows with git; an enumeration in this repository cannot receive the next entry.
    The first version of the owner listed eleven and was already missing seven of these.
    """
    names = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True, check=True
    ).stdout.split()
    assert len(names) >= 15, f"git listed {len(names)} local env vars — has the query changed?"

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import importlib.util, os, sys\n"
        f"s = importlib.util.spec_from_file_location('ge', {str(REPO / 'scripts' / 'git_env.py')!r})\n"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        "m.scrub_process_env()\n"
        "print(' '.join(sorted(k for k in os.environ if k.startswith('GIT_'))))\n"
    )
    env = {**GIT_ENV.without_git_leaks(), **dict.fromkeys(names, "/decoy")}
    done = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, check=True, env=env)
    left = set(done.stdout.split())
    assert not left - GIT_ENV.KEEP, f"survived the scrub and is not on the allowlist: {sorted(left - GIT_ENV.KEEP)}"


def test_the_script_still_imports_under_the_gates_own_interpreter_flags(tmp_path):
    """`local_ci.sh` prefixes `PYTHONSAFEPATH=1` onto the suite command — not an `export`, so it
    reaches pytest and its children but not the gate's own direct calls to this script. Under it the
    interpreter does not prepend a script's own directory, so a sibling `import` inside `scripts/`
    resolves when run by hand and raises `ModuleNotFoundError` under the suite. Fifteen existing
    tests were green in isolation and red in the full run for exactly that reason.
    """
    done = subprocess.run(
        [sys.executable, str(CITATIONS), "--self-test"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**GIT_ENV.without_git_leaks(), "PYTHONSAFEPATH": "1"},
    )
    assert done.returncode == 0, "the script does not run under the flags the suite uses:\n" + done.stdout + done.stderr
