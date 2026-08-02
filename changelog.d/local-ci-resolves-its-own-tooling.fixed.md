`scripts/local_ci.sh` now resolves its own interpreter and `ruff` instead of inheriting an
activated conda environment. As the pre-push hook, pre-commit invoked it with its own `PATH`,
so `python` and `ruff` did not exist and every check failed with `command not found` while the
script printed a per-check `FAIL` and `GATE RED -- do not push` -- indistinguishable at a glance
from a real red gate on content. The hook could therefore never pass.

An environment failure now exits 2 with `GATE CANNOT RUN` and says that nothing was measured,
rather than reporting it as a code failure.

An explicitly set `MFG_PYTHON` is honoured or the gate refuses; it is never searched past, so the
gate cannot silently run against an interpreter the operator did not name. Candidate interpreters
must import `mfgarchon`, `pytest` and `xdist` **from outside the source tree** and echo back a
token: an in-tree probe passes on any interpreter that merely has the third-party dependencies
(cwd is on `sys.path`, and cwd is the repo root, which contains the package), and an
exit-status-only probe accepts `/bin/echo` as an interpreter.

Every run now prints the interpreter and ruff version it used, so the pasted gate output that
serves as merge evidence states what was actually measured.
