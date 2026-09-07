Restored four tests removed by the #2227 suite reset, each admitted under the class that actually
fits it rather than the class it was removed from. #1564/#1560 (RFC #1574 Phase 0 guards) and #1910
(GPU particle absorbing-BC refusal) return as class-3 defect pins carrying retirement conditions
that fire when the refused capability lands, with the failure message saying to delete the pin
rather than restore the raise. #1941 (`get_active_set` vs `is_feasible`) returns as a consistency
assertion over the relation between the two predicates, replacing hard-coded tolerance verdicts.
#1783 (the `**kwargs` volatility-field gate) returns under class 1 on six measured mutations; its
call-site-counting test was not restored, the counts being a function of the branch tip.
