- **The suite's warnings are ratcheted on identity** (Issue #2119). #2118 stopped the gate printing
  pytest's 6,030-line warnings summary — 95.7% of everything it emitted — because that volume broke
  pre-commit's writer. **Suppressing a listing makes ignoring it cheaper, so on its own that was a
  regression in attention, not a fix.** "The count is still in the tail" is not a defence: a number
  scrolls past exactly the way 6,030 lines did, and the evidence is that the listing was printed in
  full every run for a year while none of the 456 deprecated calls it reported were retired.

  `tests/conftest.py` now writes a census during the gate's own suite run — no second run, and no
  per-worker merge, because the controller's `terminalreporter.stats["warnings"]` is complete under
  `-n auto` and `--disable-warnings` alike (measured, 40 of 40 either way). `check_warnings.py`
  ratchets it bidirectionally against `scripts/warning_baseline.json`: **225 identities**, a new one
  is a regression, a vanished one is progress that must be recorded.

  **Keyed on identity, not count, and the key was chosen by measurement after two designs were
  falsified by the next sample:**

  | key | run 1 | run 2 | run 3 |
  |:--|--:|--:|--:|
  | occurrences | 5021 | 5022 | 5022 |
  | raw `text[:60]` | 315 | 318 | — |
  | digits→`N`, `text[:60]` | 240 | 240 | **230** |
  | digits→`N`, `text[:40]` | **225** | **225** | **225** |

  Occurrences jitter, so an exact gate flakes and a banded one lets new warnings in silently. Raw
  text embeds measurements (`Hybrid neighborhood: 4/21 points (19.0%)`), so each count was its own
  identity. The 60-character key was called stable on two agreeing runs and the third broke it.

  **40 is stable by being coarser, not by removing the variation, and that costs something
  measurable**: against the 60-character key it merges 5 groups, about three of them real
  distinctions — `signature 'legacy'` with `'neural'`, and Newton's "iteration budget" with
  "residual stopped decreasing". A warning differing from an existing one only past character 40 of
  the same file and category raises no new identity. Stated in `conftest.py` rather than assumed.

  Dependency warnings are normalised to `site-packages/<pkg>/…` and `stdlib/…` rather than dropped
  or left absolute: 7 of the 225 sit outside the repo, and their absolute paths carried this
  machine's conda prefix into a committed baseline.

  `--self-test` in the family convention, and `check_warnings` is added to the gate's self-test
  enumeration — a ratchet outside the mechanism that checks ratchets is the gap that step exists to
  close. Verified: breaking the comparison so it always reports OK turns both directions of the
  self-test red.
