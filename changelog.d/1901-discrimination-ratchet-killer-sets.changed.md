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
