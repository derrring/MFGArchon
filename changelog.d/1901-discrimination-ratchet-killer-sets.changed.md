- **The discrimination ratchet compares WHICH tests kill each mutation, not only how many** (#1901).
  A kill count cannot see a one-for-one swap: one mutation held at **19 → 19** while one test stopped
  noticing the convention and a different one started, and the gate reported no change. The killer
  sets already existed in `discrimination_killmatrix.json`, committed beside the baseline — but
  nothing read them. Evidence with no reader, inside the ratchet that exists to catch exactly that.

  `--check-baseline` now loads the sibling matrix and reports any killer that **left**, even when the
  count holds or rises: two arriving and one leaving is a net gain by count and a real loss by
  coverage. Arrivals are reported too, on the same contract as the counts — otherwise the next
  baseline encodes the gain as if it had always held.

  Three failure modes closed with it. A baseline mutation absent from the matrix was silently
  unchecked, so the comparison now reports `compared N of M`. The absent-matrix path printed a note
  and exited 0, invisible in a three-hour log; it exits **2**, the "could not measure" code. And the
  node-ID parser truncated at the first whitespace, so parametrisations sharing a prefix collapsed
  into one set member — harmless while both sides truncated identically, not harmless once the
  strings are compared as identities.
