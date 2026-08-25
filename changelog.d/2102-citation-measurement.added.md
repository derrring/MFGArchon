- **`scripts/check_citations.py` — measures `path.py:NNN` citations in prose that no longer point at
  what they name** (Issue #2102). A line number in a document is a claim with an expiry date, and
  nothing marked it when it stopped being true. Measured on this repository: **19 of 39 adjudicable
  citations (49%) name a symbol that is not within 12 lines of the cited line**, worst in
  `mfgarchon/` source comments — the class outside every checker in `scripts/`. That percentage is a
  function of the window and has no plateau (64% at 1 line, 23% at 200); the window-independent
  floor is 9 of 39, whose symbol is not in the target file at all. Checking only "is the line inside
  the file" is nearly useless: of 226 citations exactly one points past EOF. A citation is
  adjudicable exactly when its own line names a symbol — nothing is borrowed from a neighbouring
  line — and the other 154 are **recorded as unadjudicable, never counted as passing**. Measurement
  only: no baseline, exits 0 whatever it finds, exit 2 when the instrument cannot report. The
  ratchet is a separate change on purpose.
