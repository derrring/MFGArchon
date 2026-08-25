- **`scripts/check_citations.py` — measures `path.py:NNN` citations in prose that no longer point at
  what they name** (Issue #2102). A line number in a document is a claim with an expiry date, and
  nothing marked it when it stopped being true. Measured on this repository: **33 of 73 adjudicable
  citations (45%) name a symbol that is not within 12 lines of the cited line**, worst in
  `mfgarchon/` source comments — the class outside every checker in `scripts/`. Checking only "is the
  line inside the file" is nearly useless: of 212 citations exactly one points past EOF. A citation
  is adjudicable exactly when the prose names a symbol beside it; the other 120 are **recorded as
  unadjudicable, never counted as passing**. Measurement only — no baseline, exits 0 whatever it
  finds, exit 2 when the instrument cannot report. The ratchet is a separate change on purpose: an
  instrument that will gate merges should first be read on numbers nobody has to act on.
