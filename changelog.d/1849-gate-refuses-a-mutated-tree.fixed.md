- **The gate refuses to report on a tree a killed discrimination sweep left mutated** (Issue #1849).
  `scripts/test_discrimination.py` edits production source in place and restores it in a `finally`,
  which survives an exception and SIGINT but neither SIGKILL nor a harness timeout. Its own
  `_assert_clean_tree()` runs at *its* startup, so it protects the next sweep and nothing else —
  a gate, a solve or a commit in between proceeds on mutated source.

  Observed twice. The second time was in the main checkout, and the leftover was `hjb_residual_norm`
  with its load-bearing `sqrt(dx)` deleted — the exact convention two tests written that hour existed
  to guard, so the visible symptom would have been "the new tests are wrong".

  `scripts/local_ci.sh` now greps `mfgarchon/` for the `# MUTATED` marker before any check and exits
  through `cannot_run` (**exit 2**, `GATE CANNOT RUN`) rather than as a red gate, because nothing was
  measured about the code under test. The refusal names the file, the line and the recovery command.

  Complete rather than partial, and measured: all 24 mutations carry the marker, asserted by
  `test_every_mutation_carries_the_marker_the_guard_greps_for` so a future axis added without one
  cannot silently fall outside the guard. `test_the_guard_actually_refuses` plants a marker, runs the
  gate and requires exit 2 — behavioural, not a grep over the script's own text.
