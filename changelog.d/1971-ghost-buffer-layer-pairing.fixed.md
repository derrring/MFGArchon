`GhostBuffer` paired ghost and interior layers in opposite directions at `ghost_depth > 1`, on
both walls — the third and last copy of the reversal corrected in #1967. The ghost slice `[0:g]`
runs outermost-first while `[g:2g]` runs nearest-first, so the element-wise assignment gave ghost
layer 1 the value computed for the farthest interior cell; the high side had the mirror problem.
At `ghost_depth = 1` a one-element slice is its own reverse, which is why the only depth anything
uses looked correct. (#1971)
