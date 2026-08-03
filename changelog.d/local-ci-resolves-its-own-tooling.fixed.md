`scripts/local_ci.sh` now resolves its own interpreter and `ruff` instead of inheriting an
activated conda environment. As the pre-push hook, pre-commit invoked it with its own `PATH`,
so `python` and `ruff` did not exist and every check failed with `command not found` while the
script printed a per-check `FAIL` and `GATE RED -- do not push` -- indistinguishable at a glance
from a real red gate on content. The hook could therefore never pass.

An environment failure now exits 2 with `GATE CANNOT RUN` and says that nothing was measured,
rather than reporting it as a code failure.

An explicitly set `MFG_PYTHON` is honoured or the gate refuses; it is never searched past, so the
gate cannot silently run against an interpreter the operator did not name. Candidates must import
the **tooling a run actually uses** -- `yaml` for `--fast`, plus `pytest` and `xdist` for a full
run -- **from outside the source tree**, and echo back a token: an in-tree probe passes on any
interpreter that merely has the third-party dependencies (cwd is on `sys.path`, and cwd is the
repo root, which contains the package), and an exit-status-only probe accepts `/bin/echo`.

The probe never requires the package under test. Under an editable install `import mfgarchon`
loads the tree being reviewed, so gating on it turned a broken `__init__` into `GATE CANNOT RUN
-- not a code failure`, which is precisely backwards. It is a preference when choosing between
interpreters, never a reason to refuse; if the package does not import, the checks that need it
report it, with the traceback, under a `GATE RED`.

Every run prints the interpreter and ruff version it used, at the head and again beside the
verdict, so the pasted tail that serves as merge evidence states what was actually measured.

`--fast` no longer requires the test tooling it does not invoke, and the probe covers `yaml`,
which the workflow-integrity step needs and which is not a declared dependency (it arrives
transitively via omegaconf).
