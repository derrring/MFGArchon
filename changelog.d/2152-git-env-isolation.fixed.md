- **Git fixtures no longer reach the repository the environment names** (Issue #2152). `git -C <dir>`
  and `cwd=` set the working directory; neither overrides `GIT_DIR`, which names the repository and
  wins. **A `pre-push` hook run from a linked worktree has it set**, pointing at the checkout being
  pushed — measured on git 2.50.1; a push from the main checkout, or from a subdirectory of it,
  exports no `GIT_DIR` at all. The lanes under `.claude/worktrees/` are linked worktrees, which is
  why this fires for them and for nothing else, and why reproducing it from the main checkout finds
  nothing. Observed on a real push, 2026-08-28: four commits authored `t <t@t>` — a fixture's own
  `user.email` — on the branch being pushed, a `base`/`ours`/`theirs` merge, a stray `other` branch,
  every tracked file staged as deleted. Recovered from the worktree's reflog.
- **#2085 fixed this where it was found, behind a private eleven-name list, so
  `scripts/check_citations.py` and its ~20 git call sites could not reach it.** The owner is now
  `scripts/git_env.py`, with three consumers: the script, `tests/conftest.py`, and the file the list
  came from, whose copy is deleted rather than left beside the new one.
- **The list is gone; the rule is inverted.** An enumeration cannot receive the next variable, and
  git adds them: `git rev-parse --local-env-vars` already knew fifteen, seven of which the eleven
  were missing, and the config family was absent entirely. Demonstrated end-to-end through a real
  pre-push hook, *after* that version's scrub had run: a `post-commit` planted through
  `GIT_CONFIG_COUNT` + `init.templateDir` was copied into a throwaway repository by `git init` and
  **executed** during its commit — the hole `GIT_TEMPLATE_DIR` had been added to close, reached
  through the config spelling that entry's own comment names. Eight more escaped the same way,
  including `GIT_CONFIG_PARAMETERS`, which git itself sets for any `git -c … push`. So: drop every
  `GIT_*` except a named allowlist. `pre_commit.git.no_git_env` is the same shape; its allowlist is
  wider because pre-commit wants `-c` to propagate, which is what this removes.
- **Scrubbed at module level in `tests/conftest.py`, not in a fixture.** A session-scoped autouse
  fixture is instantiated at the first test *body*; measured with a probe plugin, `GIT_DIR` is still
  set through `pytest_configure`, `pytest_sessionstart`, collection and into `pytest_runtestloop`.
  No test module runs git at import today — an AST sweep over every module-level statement finds
  zero — but the argument for scrubbing the process rather than the call sites applies to collection
  as well.
- **All three consumers load the owner by absolute path.** The script reached it through
  `sys.path.append` + `import git_env`, which leaves the *name* shadowable: a decoy `git_env.py`
  earlier on `PYTHONPATH` ran instead, printed nothing unusual, and the scrub was silently skipped.
- The test asserts on the **ambient** repository, never the temporary one — with `GIT_DIR` set, a
  check on the temp tree passes because the read comes back from the same wrong place the write went
  to. It compares refs, status, `.git/config` (#2085's second symptom) and the installed hooks, and
  builds its own fixture under `isolated_env()`: `commit.gpgsign` with an unusable key,
  `init.templateDir` and `core.hooksPath` each turn a hand-scrubbed fixture into a setup error.
