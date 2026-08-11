Single-source ratchet (`scripts/check_single_source.py`, wired into `./scripts/local_ci.sh`). CLAUDE.md names three quantities that must have exactly one owner — `diffusion_from_volatility`, `fp_drift_coefficient`, and since #1894 `hjb_residual_norm` — and nothing measured whether the restatements were growing. The registry records four quantities with their current site counts (6 / 1 / 5 / 12) and fails in both directions: growth is a regression, shrink is progress that must be recorded with `--write-baseline`.

Two design points, both from measured failures:

- **Each entry carries sentinels, and a broken instrument exits 2 rather than 0.** On 2026-08-11 both `% *Nx\b` and `np\.roll\b` returned 0 hits from `git grep -E` on this machine — that grep does not implement `\b` — while the true counts were 18 and 18. Two candidate entries were one keystroke from being recorded as "already single-sourced". Every entry now declares a `sentinel_text` its pattern must match (catches a dead pattern) and a `sentinel_file` that must be scanned (catches dead globs).
- **Comments and string literals are blanked via `tokenize` before matching.** `check_fail_fast.py` records the same trap from its regex era: 40 of 164 `hasattr` "calls" were docstring mentions. Here 6 of 18 `% Nx` hits are prose about the periodic fallback.

`alg/neural` and `alg/reinforcement` are excluded per the frozen-area rule; all four entries measure 0 there today, so the exclusion changes no current count.
