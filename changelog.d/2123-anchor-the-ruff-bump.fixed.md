- **The monthly ruff bump rewrote every hook's pin, not just ruff's** (Issue #2123).
  `.pre-commit-config.yaml` carries two `rev:` lines, and `sed`'s `s///` applies once per **line**,
  so an unanchored bump also set `pre-commit/pre-commit-hooks` to the ruff version — a tag that does
  not exist, after which `pre-commit` fetches no hook environment at all and every contributor's
  `git push` is blocked.

  The `sed` is anchored to the ruff repo, and the step now re-reads the pin afterwards and fails if
  it does not read the requested version — identity rather than a changed-line count, because two
  config shapes reproduce the original defect while changing exactly one line.

  Removed with it: a second `sed` in that step and its twin in `scripts/update_ruff_version.py`,
  both rewriting a `ruff==` line that had moved out of the file they targeted. The pin's one live
  owner is `.pre-commit-config.yaml`, which `ci.yml` and `local_ci.sh` read at runtime. (`uv.lock`
  also materialises a ruff version, already three minors adrift, but nothing runs `uv sync`.)

  `update_ruff_version.py` gains a test — it had none — and `update_files` now verifies its own
  postcondition, so a bump that matches nothing raises instead of reporting success.
