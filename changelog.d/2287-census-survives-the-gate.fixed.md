The warning census is no longer destroyed by a session that measured nothing. `local_ci.sh` exports
`MFGARCHON_WARNING_CENSUS` and leaves it exported, so every later pytest invocation in the run
inherits it — and `report_discrimination.py` issues `pytest --collect-only` after the warning ratchet
has read the census, at which point the writer overwrote a real measurement with an empty one.
Measured: 225 identities to 0 identities / 294 bytes.

The gate's verdict was never affected, because it is decided before the clobber. What was lost is the
recovery the ratchet's own failure message instructs you to run: `check_warnings.py
--write-baseline` could not be fed from the artifact the gate had just produced, so re-baselining
cost a second full suite run.

`check_warnings.py`'s `MIN_TESTS` floor already refuses a census that thin; a reader-side floor
cannot prevent an overwrite that happens before it runs, so the guard is on the writer, where it
also covers the next collect-only caller rather than only the one that exposed this.
