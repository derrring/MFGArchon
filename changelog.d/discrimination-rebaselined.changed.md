Re-measured the discrimination kill matrix and baseline, stale since `f961db90` (6138 collected;
the suite is now 6384). Every one of the 24 conventions held or improved — no row lost killers.
Tests that notice at least one convention: 449 → 581; (mutation, test) pairs: 582 → 778. Also
fixed `bc_uniform_dispatch_reads_as_mixed`, whose `verify` probe had gone blind: the uniform and
per-face ghost paths now agree on Neumann bit-for-bit, so the probe could not see the mutation take
effect and the row scored INEFFECTIVE — which looks identical to "nothing catches this". Re-pointed
at Robin, where the paths still diverge widely, the row reads 43 (#2038).
