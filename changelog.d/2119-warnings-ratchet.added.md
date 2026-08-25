- **The suite's warnings are ratcheted on identity** (Issue #2119). #2118 stopped the gate printing
  pytest's 6,030-line warnings summary — 95.7% of its output BY BYTES, 94.9% by lines — because that volume broke
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

  | key | value | note |
  |:--|--:|:--|
  | occurrences | 5022 | 5021–5023 parallel, **5002 serial** — an exact gate flakes |
  | `(file, line, kind)` | 609 | stable across runs, useless across edits |
  | raw `text[:60]` | 318 | messages embed measurements, so each count is its own identity |
  | digits→`N`, `text[:60]` | 230 | **called stable on two agreeing runs; the third differed** |
  | digits→`N`, `text[:40]` | **225** | what this gates on |

  All rows computed from the same raw run so they are comparable. Stability of the shipped key:
  **eleven agreeing full-suite runs including a fully serial one**, three mine and eight an
  independent reviewer's, every one set-equal to the committed baseline.

  **What justifies 40 is samples plus a PARTIAL mechanism, and that is stated rather than dressed
  up.** The digit normalisation closes the one channel that is understood. A second channel this
  changelog originally named — a `Reason: …` suffix said to render inconsistently — was measured
  **false** by review: both forms appear in the same run, from two distinct decorated `__init__`s.
  The 60-character key's instability channel is still unexplained.

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
