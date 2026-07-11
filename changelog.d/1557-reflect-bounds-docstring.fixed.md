- **Corrected the false `_infer_reflect_bounds` docstring** (Issue #1557, partial). The FP-particle
  helper's Returns section claimed it returns "the subset of `bounds` for axes whose BC is reflective"
  and that non-reflective axes "are excluded", but the code returns ALL bounds whenever ANY segment is
  reflective (an all-or-nothing gate, not per-axis). Rewrote the docstring to describe the actual
  behavior and to mark the per-face limitation explicitly: on a mixed reflecting/absorbing domain,
  reflection ghosts are still created at the absorbing exit, mirroring mass back within ~1 bandwidth.
  The per-face numerics fix is gated with the mass-channel fix #1552 (mfg-research exp09 validation),
  since it shifts the density in the evacuation regime; only the doc is corrected here.
