Stop writing discrimination percentages into `AGENTS.md`; point at the artifact that holds the
current ones instead.

The section quoted "212 distinct tests out of 5,872 -- 3.6%", measured at `db3496f9` over the six
conventions the ratchet then tracked. The baseline now carries twenty-four conventions, the gate
prints 9.03% over 6,420 collected tests and flags that the suite has since moved again, and nothing
about the code changed between those two readings. A fraction copied into prose is stale the day the
mutation list or the suite moves, and both move.

What replaces it is the shape of the argument without the digits: most of the suite does not react
when the load-bearing physics is broken, the conventions that are defended are defended very
unevenly, and `scripts/discrimination_baseline.json` plus the gate's own output carry the numbers.
An aggregate over the whole suite cannot show a convention held by two tests, which is the thing a
reader would act on (#2148).

Two other moving figures in the same passage go the same way: a percentage of inert
single-source-shaped tests, and a reference to "six conventions" that is now twenty-four. The
historical counts stay -- #1660's breakdown and the five tests #1715 names are records of what
happened, not claims about the current tree, and they do not rot.
