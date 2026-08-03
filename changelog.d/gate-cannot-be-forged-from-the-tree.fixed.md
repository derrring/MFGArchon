The gate's two remaining `-m` invocations — `pytest` and the inline workflow-integrity check —
now run with `-P`, so the tree under review cannot supply the tools that judge it. `-m` and `-c`
put the current directory at the front of `sys.path`, and `scripts/local_ci.sh` runs from the
repo root, so a `pytest/` package or a `yaml.py` committed there shadowed the real ones.

Measured: a repo-root `pytest/` printing `1234 passed, 0 failed` was accepted as the full suite,
and a repo-root `yaml.py` returning a synthetic document made the workflow check report `PASS`
on a workflow with a dangling `needs:` — the exact failure that check exists to catch. With `-P`
both forgeries are ignored, the real pytest collects 5868 tests, and the broken workflow is
correctly reported.
