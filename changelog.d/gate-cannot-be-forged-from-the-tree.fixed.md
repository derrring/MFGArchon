The gate's two remaining `-m` invocations — `pytest` and the inline workflow-integrity check —
now run with `-P`, and the pytest step additionally with `PYTHONSAFEPATH=1`, so the tree under
review cannot supply the tools that judge it. `-m` and `-c`
put the current directory at the front of `sys.path`, and `scripts/local_ci.sh` runs from the
repo root, so a `pytest/` package or a `yaml.py` committed there shadowed the real ones.

Measured: a repo-root `pytest/` printing `1234 passed, 0 failed` was accepted as the full suite,
and a repo-root `yaml.py` returning a synthetic document made the workflow check report `PASS`
on a workflow with a dangling `needs:` — the exact failure that check exists to catch. With `-P` the workflow forgery is ignored and the broken
workflow is correctly reported.

`-P` alone is not sufficient for pytest: it is per-process, and `pytest-xdist`'s execnet workers
do not inherit it, so under the gate's own `-n auto` a repo-root `pytest/` still reaches them.
Measured on a probe test that must fail: `-P` alone crashes every worker (`no tests ran`);
`PYTHONSAFEPATH=1` in addition lets the real pytest run and report `1 failed`. Anything that forks
needs the environment variable, not just the flag — which is the property a future lint assertion
over this script should check, rather than merely "every `-m` carries `-P`".
