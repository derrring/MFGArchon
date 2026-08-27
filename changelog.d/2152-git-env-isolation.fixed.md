- **Git fixtures no longer write into the repository `GIT_DIR` names** (Issue #2152). `git -C <dir>`
  and `cwd=` both set the working directory; neither overrides `GIT_DIR`, which names the
  repository and wins. Git exports it to every hook, so the suite invoked from the pre-push hook
  inherits one pointing at the checkout being pushed — and every fixture building a throwaway
  repository operated there while looking isolated at each call site. Observed on a real push,
  2026-08-27: four commits authored `t <t@t>` (the fixture's own `user.email`) landed on the branch,
  a `base`/`ours`/`theirs` merge and a stray `other` branch appeared, every tracked file was staged
  as deleted, and three fixture directories were left in the tree. Recovered from the reflog.
- **#2085 fixed this where it was found and nowhere else, and that is the shape of this issue.**
  The variable list was private to `tests/unit/test_prune_local_branches.py`, so
  `scripts/check_citations.py` and its ~20 git call sites could not reach it. The list now has one
  owner, `scripts/git_env.py`, with three consumers: the script, `tests/conftest.py`, and the file
  it came from, whose private copy is deleted. Two of its eleven entries were added only after the
  first fix proved insufficient — a copy made at that moment would still be missing them.
- **Scrubbed at the process level, not threaded through call sites.** An `env=` on each
  `subprocess.run` is correct only at the sites someone edited; the failure mode is the next fixture,
  written by someone with no reason to know any of this. One autouse session fixture in
  `tests/conftest.py` and one call at the top of `check_citations.py` cover every call site
  including the ones added later — and including that script's measurement path, where a leaked
  `GIT_DIR` meant `git -C <root> ls-files` read a repository nobody named.
- The global and system config are deliberately left alone on paths that read a real repository:
  blanking them changes how git resolves `safe.directory`. `isolated_env()` adds that blanking and
  is used only for throwaway repositories.
- `tests/unit/test_git_fixtures_do_not_touch_the_ambient_repo_2152.py` asserts on the *ambient*
  repository, never on the temporary one. Asserting the fixture worked is what passing looks like
  when the isolation is gone: with `GIT_DIR` set, the read comes back from the same wrong place the
  write went to.
- The script reaches its shared module through an appended `sys.path` entry, not a plain sibling
  import. The gate exports `PYTHONSAFEPATH=1`, every subprocess inherits it, and under it the
  interpreter does not prepend a script's own directory — so the plain import resolved when the
  file was run by hand and raised `ModuleNotFoundError` under the gate. Fifteen tests passed in
  isolation and the same fifteen were red in the full suite. Pinned by
  `test_the_script_still_imports_under_the_gates_own_interpreter_flags`, because nothing about
  running the affected tests locally can show it.
