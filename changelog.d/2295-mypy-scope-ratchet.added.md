`scripts/check_mypy_scope.py` records one type-error count per top-level package and fails the gate
when any of them moves, in either direction. The gate's own mypy step checks `mfgarchon/config` —
1 of 14 subpackages, scope copied from `ci.yml` — and this does not widen it; what it adds is a
number for the other thirteen, which had none, so nothing said whether they were getting worse.
Baseline at writing: 1304 errors over 10 measured packages, 274 source files checked.

The half that is not a count: four packages (`utils`, `workflow`, `visualization`, `backends`) are
dropped by `[tool.mypy] exclude` and are not scanned at all. `mypy mfgarchon/utils` prints "There
are no .py[i] files in directory", not a zero — so a census that greps for "Found N errors" and
defaults to 0 reports them clean. Measured while writing this: that reading gave five clean packages
where one is clean and four are unmeasured. Every count is therefore stored beside a status, and a
package that leaves the scan fails the ratchet instead of reporting zero.

The script's own control caught the script's own defect. mypy emits `path:LINE:COLUMN: error:` under
this repo's `show_column_numbers`, and a pattern written for `path:LINE: error:` matched 23 of 1304
lines — a small, plausible number that would have pinned a baseline blind to the other 1281. `scan()`
now cross-checks the parser against a raw substring count of the same output and refuses to report
when they disagree, and the self-test exercises `attribute()` on both line formats rather than only
the comparison it feeds.
