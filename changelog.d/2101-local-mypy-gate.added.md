The blocking mypy type gate ci.yml runs on `mfgarchon/config` now runs in `./scripts/local_ci.sh`
too, so a type error in that subpackage is visible before pushing rather than only on GitHub.
~22 s on a cold `.mypy_cache`, ~0.6 s warm. An interpreter without mypy is skipped during
interpreter selection rather than aborting the gate, and the step carries a positive control -- a
file that must fail to type-check -- so a mypy that has been silenced by a config edit or a
disabled error code is reported as an instrument failure instead of passing as clean code.

Environment faults are separated from code faults by mypy's OUTPUT, not its exit code: a missing
dependency that `pyproject.toml` declares, or a plugin that fails to load, reports as an environment
failure; everything else, including the exit-2 that an emptied or renamed target produces, reports
as a normal gate failure with the code attribution intact.
