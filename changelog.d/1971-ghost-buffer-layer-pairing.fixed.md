`GhostBuffer` paired ghost and interior layers in opposite directions at `ghost_depth > 1`, on both
walls. The ghost slice `[0:g]` runs outermost-first while `[g:2g]` runs nearest-first, and the
element-wise assignment gave ghost layer 1 the value computed for the farthest interior cell; the
high side had the mirror problem. At `ghost_depth = 1` a one-element slice is its own reverse,
which is why the only depth anything uses looked correct.

This is the copy in `GhostBuffer._update_bounded` and it does **not** close the family: the same
slice-pair construct is written out twice more in `PreallocatedGhostBuffer`, both reversed at both
walls, tracked as #1966 Defect 2. (#1971)
