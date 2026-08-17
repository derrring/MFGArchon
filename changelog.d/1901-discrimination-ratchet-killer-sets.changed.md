- **The discrimination ratchet now compares WHICH tests kill each mutation, not only how many.**
  A kill count cannot see a one-for-one swap: `drift_coefficient_2x` held **19 → 19** while one test
  stopped noticing the convention and a different one started, and the gate reported no change
  (#1901). The killer sets already existed — `scripts/discrimination_killmatrix.json`, committed
  beside the baseline with a test pinning the two to the same run — but **nothing read them**. They
  were evidence with no reader, which is the same shape the ratchet exists to catch, sitting inside
  the ratchet.

  `--check-baseline` now loads the sibling matrix and reports any killer that **left**, even when
  the count holds or rises. Two arriving and one leaving is a net gain by count and a real loss by
  coverage. Arrivals alone are reported as an improvement to record, same contract as the counts:
  otherwise the next baseline encodes the gain as if it had always held.

  Renames land here too, and that is the intended cost — the fix is to regenerate the baseline in
  the same commit, exactly as a count change already requires. If the matrix is absent the gate
  degrades to counts **and says so on stdout**, because a silently weaker gate is the failure mode
  this tool is for.

  Mutation-verified: neutering the killer-set check reddens 2, ignoring departures reddens 2,
  dropping the `INEFFECTIVE` skip reddens 1. The new tests include a control that identical sets
  report nothing, so the check cannot pass by being noisy.

- **Why this before anything else.** A corpus-wide mining pass over every issue and all ~110
  `changelog.d/` post-mortems (#1901) ranked the defect classes that actually recur, by confirmed
  instances: guards green for the wrong reason **17**, instruments reporting a verdict without their
  denominator **16**, insensitive oracles **12** — against **5** for the numerics class. Three of the
  top three are defects in the *measurement*, not the product. Of 5,665 collected tests, **5,481
  (96.8%) notice nothing** when any of the six load-bearing conventions is broken. The target is the
  discriminating fraction, which this ratchet is the only thing that measures.

- **Review round — a production-fatal blocker the new tests structurally could not see.**
  `main()` writes the killer list under `"killed"`; the comparison read `"failed"`. On every real
  run `after` was empty, so **all 220 baseline killers reported as departed on an unchanged tree**,
  and the next weekly sweep would have burned ~3 h of runner time to file a false
  *"Weekly test-discrimination sweep is failing"*. Nothing pre-merge catches it: `local_ci.sh` runs
  four other ratchets, not `--check-baseline`.

  The six new tests all passed, because their fixture **fabricated `"failed"` too** — it carried its
  own copy of the producer's data shape, so it could only prove self-consistency. The file's own
  pre-existing helper uses `"killed"`. The mutation table shipped with the PR (2/2/1) was true and
  worthless: satisfied by an implementation that is unconditionally wrong.

  No value-level test can catch this, so the guard is structural —
  `test_the_reader_uses_the_key_main_writes` asserts against the source of *both* sides, plus
  `test_end_to_end_an_unchanged_tree_reports_nothing`, which replays the committed matrix as if the
  sweep had just produced it. Reverting the key now reddens **4**.

- **Four more findings from the same review, all taken.** Arrivals were suppressed whenever anything
  departed — exactly the rename case, where the arrival is the one fact separating "regenerate" from
  "investigate". A baseline mutation absent from the matrix was silently unchecked (a genuine swap
  returned no problems and no message), so the comparison now reports `compared N of M`. The absent-
  matrix path printed a NOTE and exited 0, invisible in a three-hour log; it exits **2** now, the
  "could not measure" code. And the failure text said "regenerate with `--write-baseline`", which
  leaves the matrix stale, reddens the pinning test and costs a second sweep — it now gives the
  single invocation that regenerates both.

- **The node-ID parser truncated at the first whitespace.** `(\S+?)` cut
  `test_x[a-V+f(m), lambda=2]` at the comma; two committed killer IDs were already truncated.
  Harmless while both sides truncated identically, not harmless once these strings are compared as
  identities — three parametrisations sharing a prefix collapse into one set member. Fixed to
  `(.+?)(?: - |$)`, and the two affected entries repaired **mechanically**: each truncation resolved
  to exactly one collected node ID out of 6,163, so the repair is determined rather than guessed.

- **One survivor was deleted rather than pinned.** `if not before and not after: continue` is a pure
  no-op — with both sets empty neither branch fires. The right answer to "no test catches this" is
  deletion, not a test for a no-op. Two message cosmetics (the held/moved word, the `+N more`
  truncation) stay deliberately unpinned and the docstring now says so, since they change no verdict.

- **The premise held up under measurement, which is the part worth keeping.** Review tracked node-ID
  survival across real history: over `0a5a7731 → 4c5c1840` (three weeks, 85 transitions, 294
  whole-suite IDs lost) **zero killer IDs were lost**; over the previous baseline window, exactly
  **one**, and it vanished in `0c97501a Delete 133 tests that pin the API shape` — a deletion the
  count ratchet already reported. Against that: 5 departures across the two committed matrices, 4 of
  them genuine discrimination losses, **3 invisible to counts**. Measured **4:1 signal to noise**,
  with the one noise instance already a count-ratchet true positive. Determinism across platforms:
  the committed macOS matrix against the 2026-08-10 Linux runner artifact — 0 departures, 0 arrivals.

- **#1817's "the weekly sweep had never once succeeded" is stale.** Run `31358575956`, 2026-08-10:
  **success in 2h52m** against a 300-minute budget. The 2026-08-07 failure was a true positive
  (11 → 9 LOST plus two IMPROVED), regenerated by #1851. The instrument fires, gets fixed, and
  greens — so this sharpens something live.
