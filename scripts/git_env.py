"""One owner for the git environment that defeats `-C` and `cwd=`.

`git -C <dir>` and `subprocess.run(..., cwd=<dir>)` both set the *working directory*. Neither
overrides `GIT_DIR`, which names the repository directly and wins over both. A fixture that builds a
throwaway repository with `git -C <tmp> init && git -C <tmp> commit` then operates on whatever
`GIT_DIR` names, while looking correctly isolated at every call site.

**The condition is pushing from a linked worktree.** Measured, git 2.50.1, a real `pre-push` hook
dumping its environment:

    push from the main checkout           GIT_DIR unset
    push from a subdirectory of it        GIT_DIR unset (only GIT_PREFIX=sub/)
    push from a linked worktree           GIT_DIR = <main>/.git/worktrees/<name>

Which is how the lanes in `.claude/worktrees/` work, and why this fires for anyone using one and for
nobody else. Stated as "git exports GIT_DIR to every hook" -- the first version of this file --  it
sends a reader to reproduce from the main checkout, find an empty `GIT_DIR`, and conclude the whole
thing is unnecessary. Commit-time hooks are different again: they get `GIT_INDEX_FILE` (relative)
and `GIT_AUTHOR_*`, and still no `GIT_DIR`.

That is #2085, found in `prune_local_branches`'s tests and fixed there behind a private copy of an
eleven-name list. #2152 is the same failure in `check_citations` and its tests, which the private
copy could not reach: four commits authored `t <t@t>`, a stray `other` branch and every tracked file
staged as deleted, on a real push, 2026-08-28.

## Why this is a denylist and not a list of names

The eleven-name version was itself the failure it warned about. `git rev-parse --local-env-vars`
already knew fifteen, seven of which it was missing, and the config family was absent entirely.
Demonstrated end-to-end through a real pre-push hook, after that version's scrub had run: a
`post-commit` hook planted through `GIT_CONFIG_COUNT` + `init.templateDir` was copied into a
throwaway repository by `git init` and **executed** during the fixture's `git commit` -- the exact
hole `GIT_TEMPLATE_DIR` had been added to close, reached through the config spelling that entry's
own comment names. Eight more escaped the same way: `GIT_CONFIG_PARAMETERS` (which git itself sets
for any `git -c … push`, and which reaches the hook), `GIT_CONFIG_GLOBAL`, `GIT_ATTR_SOURCE`,
`GIT_REPLACE_REF_BASE`, `GIT_DEFAULT_HASH`, `GIT_LITERAL_PATHSPECS`, and `GIT_AUTHOR_*` /
`GIT_COMMITTER_*`.

An enumeration cannot receive the next variable, and git adds them. So the rule is inverted: drop
everything `GIT_*` except what must survive. `pre_commit.git.no_git_env` is the same shape and is
installed in this environment; its allowlist is wider because pre-commit *wants* `-c` to propagate
across its own subprocesses, which is the thing being removed here.
"""

from __future__ import annotations

import os

#: The only `GIT_*` a throwaway repository needs. `GIT_EXEC_PATH` is where git's own subcommands
#: live and removing it breaks git itself; the rest are transport credentials, which no fixture
#: uses but which cost nothing to keep and would be surprising to strip from a real operation.
#: Deliberately NOT here, and this is the whole point: the `GIT_CONFIG_*` family, which
#: `pre_commit`'s otherwise-identical allowlist keeps.
KEEP = frozenset(
    {
        "GIT_EXEC_PATH",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_SSH_VARIANT",
        "GIT_SSL_CAINFO",
        "GIT_SSL_NO_VERIFY",
        "GIT_ASKPASS",
        "GIT_ALLOW_PROTOCOL",
        "GIT_HTTP_PROXY_AUTHMETHOD",
    }
)


def _is_leak(name: str) -> bool:
    return name.startswith("GIT_") and name not in KEEP


def without_git_leaks(base: dict[str, str] | None = None) -> dict[str, str]:
    """`base` with every inherited `GIT_*` dropped, so `-C` and `cwd=` mean what they say.

    This is the form for reading a repository that really exists. It leaves the user's global and
    system config *files* in place -- it removes only what the environment overrode. A measurement
    over someone's own checkout should resolve aliases, `core.excludesFile` and diff settings the
    way their git does; blanking those is a fixture's need, not a reader's.
    """
    source = os.environ if base is None else base
    return {k: v for k, v in source.items() if not _is_leak(k)}


def isolated_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """`without_git_leaks`, plus no config from anywhere -- the form for a throwaway repository.

    A fixture repository sets whatever `user.*` it needs on itself. Reading the developer's global
    config makes verdicts depend on it, and three ordinary settings turn a fixture into an error
    rather than a failure: `commit.gpgsign` with a key this process cannot use, `init.templateDir`
    and `core.hooksPath` pointing at hooks that run during the fixture's own commit.
    """
    env = without_git_leaks(base)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    return env


def scrub_process_env() -> None:
    """Drop the inherited `GIT_*` from this process, so no call site has to remember.

    Preferred over threading `env=` through every `subprocess.run`: a per-call scrub is correct only
    at the sites that were edited, and the failure mode is a NEW call site written later by someone
    with no reason to know any of this. Config is left alone for the reason `without_git_leaks`
    gives; a caller that wants a fixture's isolation asks for `isolated_env` explicitly.

    One-shot, not a guard: anything that puts a variable back with a bare `os.environ[...] = ...`
    rather than `monkeypatch` keeps it for the rest of the process.
    """
    for name in [k for k in os.environ if _is_leak(k)]:
        del os.environ[name]
