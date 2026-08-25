The blocking mypy type gate ci.yml runs on `mfgarchon/config` now runs in `./scripts/local_ci.sh`
too, so a type error in that subpackage is visible before pushing rather than only on GitHub.
~22 s on a cold `.mypy_cache`, ~0.6 s warm. An interpreter without mypy is skipped during
interpreter selection rather than aborting the gate, and a mypy that exits 2 -- a missing
dependency or plugin, nothing checked -- is reported as an environment failure rather than as a
type error in your code.
