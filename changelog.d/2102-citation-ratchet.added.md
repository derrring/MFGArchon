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
