"""One owner for the git environment variables that defeat `-C` and `cwd=`.

`git -C <dir>` and `subprocess.run(..., cwd=<dir>)` both set the *working directory*. Neither
overrides `GIT_DIR`, which names the repository directly and wins over both. Git exports `GIT_DIR`
and its relatives to every hook it runs, so anything invoked from `.git/hooks/pre-push` -- the full
test suite, in this repository -- inherits an environment pointing at the checkout being pushed.
A fixture that builds a throwaway repository with `git -C <tmp> init && git -C <tmp> commit` then
commits into the developer's branch while looking, at every call site, correctly isolated.

That is #2085, found in `scripts/prune_local_branches.py`'s tests and fixed there with a private
copy of this list. #2152 is the same failure in `scripts/check_citations.py` and its tests, which
the first fix did not reach: four commits authored `t <t@t>`, a stray `other` branch, and every
tracked file staged as deleted, on a real push, 2026-08-27.

The list lives here because a copy does not receive the next entry. Two of the eleven below --
`GIT_CONFIG` and `GIT_TEMPLATE_DIR` -- were added only after the first version proved insufficient,
and a second copy made at that moment would still be missing them.
"""

from __future__ import annotations

import os

# Every variable that can point git at a repository, an index, an object store or a config file
# other than the one the caller named.
GIT_ENV_LEAKS: tuple[str, ...] = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CEILING_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
    # `GIT_CONFIG` acts as an implicit `--file`, so `git -C <tmp> config ...` rewrites the file it
    # names -- #2085's second symptom reproduces through it even with `GIT_DIR` scrubbed.
    "GIT_CONFIG",
    # `git init` copies `$GIT_TEMPLATE_DIR/hooks/` into every repository it creates. An ambient
    # value plants hooks that then execute during a fixture's `git commit`, and blanking the global
    # config does not neutralise it -- that only covers `init.templateDir`, the config spelling.
    "GIT_TEMPLATE_DIR",
)


def without_git_leaks(base: dict[str, str] | None = None) -> dict[str, str]:
    """`base` minus the redirecting variables, so `-C` and `cwd=` mean what they say.

    This is the form for reading a repository that really exists. It deliberately leaves the user's
    global and system config in place: blanking those changes how git resolves `safe.directory`,
    and a measurement run over the developer's own checkout should not start refusing it.
    """
    source = os.environ if base is None else base
    return {k: v for k, v in source.items() if k not in GIT_ENV_LEAKS}


def isolated_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """`without_git_leaks`, plus no ambient config at all -- the form for a throwaway repository.

    A fixture repository sets whatever `user.*` it needs on itself. Reading the developer's global
    config would make the verdicts depend on it, and #2085 was in part a write that reached it.
    """
    env = without_git_leaks(base)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    return env


def scrub_process_env() -> None:
    """Drop the redirecting variables from this process, so no call site has to remember.

    Preferred over threading `env=` through every `subprocess.run`: a per-call scrub is correct
    only at the sites that were edited, and the failure mode is a NEW call site written later by
    someone who has no reason to know any of this. Config is left alone here for the reason
    `without_git_leaks` gives.
    """
    for name in GIT_ENV_LEAKS:
        os.environ.pop(name, None)
