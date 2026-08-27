Stop writing discrimination percentages into `AGENTS.md`, and say what to read instead -- including
the two ways of reading it that give different answers.

The section quoted "212 distinct tests out of 5,872 -- 3.6%" over the six conventions the ratchet
then tracked. The baseline now holds twenty-four and has been re-recorded six times in the month to
2026-08-22, so a fraction copied into prose is stale on that clock.

More to the point, the fraction mostly measures the mutation list rather than the suite. Holding the
list at the original six and recomputing against the current tree moves it by a twentieth of the
rise that came from extending the list to twenty-four.

What replaces it: the shape of the argument, a pointer to the gate's own output for the current
figures, and two cautions for a reader told to "read the vector". Summing the per-convention kill
counts in `discrimination_baseline.json` over-counts, because a test killing several mutations
appears in each -- the overlap was 8 of 220 under six conventions and is 173 of 753 now, so the sum
currently exceeds the gate's figure by three percentage points. The distinct-test number is
derivable only from `discrimination_killmatrix.json`, which the previous text pointed at and which
is now named again.

Kept, with its scope written out: the inert fraction among agreement-shaped names. Nothing in the
repo recomputes it -- the sweep's regex selects a wider population -- so it is a dated measurement
over a named set rather than a number that moves. The discriminator throughout is whether a figure
moves when the code or the suite moves, not whether it is a figure.
