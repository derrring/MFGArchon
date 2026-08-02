`scripts/local_ci.sh` now resolves its own interpreter and `ruff` instead of inheriting an
activated conda environment. As the pre-push hook, pre-commit invoked it with its own `PATH`,
so `python` and `ruff` did not exist and every check failed with `command not found` while the
script printed a per-check `FAIL` and `GATE RED -- do not push` -- indistinguishable at a glance
from a real red gate on content. The hook could therefore never pass.

An environment failure now exits 2 with `GATE CANNOT RUN` and says that nothing was measured,
rather than reporting it as a code failure. Interpreter selection requires `import mfgarchon` to
succeed, so a `python` that merely exists on `PATH` cannot be mistaken for the right environment.
