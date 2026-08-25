- **`scripts/check_citations.py` — measures `path.py:NNN` citations in prose that no longer point at
  what they name** (Issue #2102). A line number in a document is a claim with an expiry date, and
  nothing marked it when it stopped being true. Measured on this repository: **18 of 38 adjudicable
  citations (47%) name a symbol that is not within 12 lines of the cited line**, worst in
  `mfgarchon/` source comments — the class outside every checker in `scripts/`. That percentage is a
  function of the window and has no plateau (63% at 1 line, 21% at 200); 8 rows survive any window —
  7 whose symbol is not in the target file at all, plus one past EOF. Checking only "is the line
  inside the file" is nearly useless: exactly one citation in the repository points past EOF. A
  citation is adjudicable exactly when **its own line** names a symbol — nothing is borrowed from a
  neighbour — and the rest are **recorded as unadjudicable, never counted as passing**. Measurement
  only: no baseline, exits 0 whatever it finds, exit 2 when the instrument cannot report — which it
  does over an unmerged index, over a failed index query, and when two index entries name one file.
