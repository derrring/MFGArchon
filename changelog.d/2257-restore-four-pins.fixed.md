Restored four tests removed by the #2227 suite reset, each admitted under the class that actually
fits it rather than the class it was removed from. #1564/#1560 (RFC #1574 Phase 0 guards) and #1910
(GPU particle absorbing-BC refusal) return as class-3 defect pins carrying retirement conditions
that fire when the refused capability lands, with the failure message saying to delete the pin
rather than restore the raise. #1941 (`get_active_set` vs `is_feasible`) returns as a consistency
assertion over the relation between the two predicates, replacing hard-coded tolerance verdicts,
plus the one assertion those masks carried that was not a tolerance verdict: a point exactly at the
bound is binding. #1783 (the `**kwargs` volatility-field gate) returns under class 1 on eight
measured mutations, and now pins the refusal on the Newton path as well as the Picard one, which no
test in the repository did before.

Two things a class-3 pin cannot infer are now asserted rather than assumed: that the guarded path
was actually reached, since "nothing was raised" also happens when a dispatch routes around the
guard; and, in the retirement text, what was observed rather than that the capability landed.
