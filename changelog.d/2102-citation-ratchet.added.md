Ratcheted citation drift. `scripts/check_citations.py --check-baseline` now runs in
`./scripts/local_ci.sh`, refusing a change that adds a `path.py:NNN` citation whose named symbol is
no longer near the cited line. The baseline records **which** claims are drifted -- keyed on the
resolved target file, the cited line, and the symbols the prose names beside it -- not only how
many: pinning counts alone is satisfied by hiding two citations while adding two, and pinning
identities alone is satisfied by a second citation to a line already recorded. Both are pinned, in
both directions, alongside `adjudicable` against shrinking, since deleting the symbol name from
beside a citation moves it to `unadjudicable` where nothing judges it. A version bump trips this by
design -- collating `changelog.d/` into the exempt `CHANGELOG.md` moves rows -- which is step 3 of
the version-bump checklist in `AGENTS.md`.

  The recorded set is a **review queue, not a defect list**. All 18 rows were read by hand: one is a
  withdrawn claim (#2112, and neither of the other two classes). The remaining 17 contain both
  genuinely wrong citations and cases of the instrument attributing a symbol from a different clause
  on the same line -- **and no proportion between them is worth acting on.** Two independent
  hand-reads returned mirror-image splits, so the classes are hard to tell apart even by hand; a
  binomial interval on seventeen rows spans 29% to 76% either way. Character distance between
  citation and symbol was measured as a mechanical discriminator and does not separate them either
  (median 8 for real against 9 for artifact). So the gate's job is that a new row gets looked at,
  not that every row is a defect.

  The gate's own red message carries the operational copy of this and is the one to change if the
  numbers move; this paragraph is the release-note copy and cites #2112 for the withdrawn row.
