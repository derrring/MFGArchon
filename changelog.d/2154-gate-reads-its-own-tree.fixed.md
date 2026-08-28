- **The gate reads the tree it is gating, says which one, and refuses if it is the wrong one**
  (Issue #2154). `"$PY" scripts/X.py` puts `scripts/` on `sys.path[0]`, not the repository root;
  `scripts/` holds no `mfgarchon`, so the import falls through `PathFinder` to setuptools' editable
  finder, whose mapping is hard-wired to the original checkout. From a `git worktree` the capability
  matrix and the deprecation self-test therefore measured a different tree, on whatever branch it
  happened to be sitting on. Observed 2026-08-28: three capability cells `UNSUPPORTED -> FAIL` and
  `GATE RED` on a branch none of the changed code belonged to — the same commit passed with
  `PYTHONPATH` set and failed without it, one minute apart.
- **Exported once, not passed at each call site.** `scripts/local_ci.sh` sets `PYTHONPATH="$PWD"`
  right after it `cd`s to the root, so every script invocation gets it, including ones added later.
  `_EditableFinder` sits after `PathFinder` in `sys.meta_path`, so the entry wins, and `PYTHONPATH`
  survives both `-P` and `PYTHONSAFEPATH=1`, which the suite step needs.
- **`gate package : <path>` joins `gate interpreter` in the head and in the pasted tail**, and the
  gate now exits before any check if that path is not under the tree it is gating. #2146 established
  the same fact and put it in `AGENTS.md`, where nothing checks it against a run; a gate that
  imported a tree it is not gating should not return a verdict at all.
- Two details the mutation caught, both of which would have made the new line decorative. The probe
  runs under `-P`: without it, `python -c` puts CWD on `sys.path[0]`, CWD is the repository root, and
  the probe reports the local package however broken the resolution is for the scripts — a proxy
  that cannot fail. And the comparison uses `pwd -P` rather than `$PWD`, because the probe reports a
  `resolve()`d path with symlinks followed while `$PWD` is the logical path the caller arrived by;
  compared unresolved, any repository reached through a symlinked parent fails on a correct run.
  Controlled through a real symlink: no false refusal, and the direct path still passes.
- **Recorded, not fixed here:** at least seven scripts import the package and the gate invokes two.
  The others — `audit_deprecated_symbols.py`, `check_circular_imports.py`,
  `generate_deprecation_guide.py`, plus `capability_census.py` (`importlib.import_module`) and
  `run_health_check.py` (`exec("import mfgarchon")`), neither of which any import-shaped text or AST
  scan can see — still read the main checkout when run by hand from a worktree. My first count said
  five and missed exactly the two `AGENTS.md` warns about eight lines above the paragraph this PR
  rewrote.
  They produce reports rather than verdicts.
