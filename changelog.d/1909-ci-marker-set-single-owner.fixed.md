- **The gate now honours `manual`, and its marker set has one owner** (Issue #1909).
  `pytest.ini` declares `manual` as *"Runs in NO automatic tier"*, and `scripts/local_ci.sh` did
  not exclude it. The contract held only by coincidence — all ten `manual` tests happened to also
  carry `slow`, which `not slow` removed — so marking a test `manual` without `slow` would have
  run it in the authoritative gate, silently, which is the opposite of what the marker promises.

  The set was also written twice, byte-identical, in `local_ci.sh` and `scripts/test_discrimination.py`,
  bound by a comment saying they must match. A comment is not a mechanism: diverging them measures
  every kill count in `discrimination_baseline.json` against a different population than the gate
  runs, and nothing reports it. Both now read `scripts/ci_markers.txt`.

  `tests/unit/test_ci_marker_set.py` keeps it fixed, and it discriminates — measured: removing
  `not manual` from the file turns it red. The non-automatic markers are read from `pytest.ini`'s
  own declarations rather than hard-coded, so a future marker with the same contract is covered
  the day it is declared.

  The recorded population is unchanged: 6002 collected and 388 deselected under both the old and
  the new marker set, so the existing kill counts remain valid without a re-sweep.
