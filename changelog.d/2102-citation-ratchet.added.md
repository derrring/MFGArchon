Ratcheted citation drift. `scripts/check_citations.py --check-baseline` now runs in
`./scripts/local_ci.sh`, refusing a change that adds a `path.py:NNN` citation whose named symbol is
no longer near the cited line. It pins two numbers: `drifted` bidirectionally, and `adjudicable`
against shrinking -- deleting the symbol name from beside a citation would otherwise lower `drifted`
and read as progress while moving the row to `unadjudicable`, where nothing judges it.
