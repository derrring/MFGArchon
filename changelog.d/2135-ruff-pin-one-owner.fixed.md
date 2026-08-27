- **The ruff pin has one reader and one writer** (Issue #2135). One version was read by five
  different expressions across `scripts/local_ci.sh` and two workflows — four of them a hand-rolled
  `grep -A1 astral-sh/ruff-pre-commit` — and written by two, `update_ruff_version.py` and the
  workflow's own `sed -i`. They disagreed on what a ruff block may look like, so the writer could
  land a bump the verifiers then read as absent; #2123 had to be fixed once on each. Reading is now
  `--print-current` and writing is `--force`, and every call site goes through them.
  `--print-current` prints the version and nothing else, before the banner and before any network
  call, with diagnostics on stderr so a caller reading `$(...)` cannot capture an error message as
  a version.
- **Consolidating six expressions into one silently dropped two checks, and both are restored.**
  The surviving reader accepted anything made of digits and dots, where `ci.yml`'s deleted
  expression had required three components — so `rev: v.` printed `.` into a `pip install ruff==`
  whose only guard is `-z`. And anchoring the pin expression to the ruff `repo:` line, while
  correct on the axis #2123 is about, refused four shapes the old greps read: a quoted URL, a
  `.git` suffix, a trailing comment on the `repo:` line, and `http://`. All four are ordinary YAML
  that `pre-commit` accepts. `_require_version` is now the one place that says what a version is,
  used by the reader and by `--force` — which validates before writing, so `--force abc` no longer
  leaves `rev: vabc` on disk and `--force '\g<0>'` no longer expands as a regex backreference.
- **The one-owner test asserts the property rather than the previous bug's spelling.** Its first
  version searched three named files for `grep -A1 …ruff-pre-commit`; independent review wrote
  twelve second implementations that all passed it — `grep -A2`, `awk`, `yq`, `perl -pi`,
  `python -c`, a `sed -i` reached through a variable, `sed … > tmp && mv tmp config`, and the same
  expressions placed in another workflow or a new script. It now scans everything under `scripts/`
  and `.github/` for lines that name where the pin lives *and* pipe it through a text tool or edit
  it in place, with an explicit allowlist and a sentinel that fails when the scan stops selecting
  files.
