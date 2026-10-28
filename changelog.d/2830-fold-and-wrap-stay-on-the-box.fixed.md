- **The particle fold and wrap keep a position on the box at either wall** (#2830). Rounding took both off it:
  - `corner.reflect_positions` mapped 0.1 on [-0.3, 0.1] to 0.10000000000000003, because fl(0.1 + 0.3) rounds above
    the span;
  - `periodic.wrap_positions` mapped the float64 just below -0.9 on [-0.9, 0.2] to 0.20000000000000007, against its
    promise of [xmin, xmax). A float32 array, which it computes in float64 and rounds once, came back at float32(xmax);
  - the 1-D GPU helper's fold landed in float32 up to 9.5e-8 below the wall 0.1 on [0.1, 1.1], and on MPS, where
    torch's float32 remainder can be negative, its wrap landed below xmin.

  Each rule now has one per-axis implementation that the owner and the GPU helper both call, `corner.fold_axis` and
  `periodic.wrap_axis`, on NumPy or on a backend's array module. The fold's result lies in [xmin, xmax] and the wrap's
  in [xmin, xmax), as the result's dtype holds them; on a box that is one point in that dtype, the wrap returns the
  point. Only rounding residue is moved: the fold clips it onto the wall, and the wrap takes a result at or past xmax
  to the largest value below it and one below xmin to xmin. A move larger than the rule's own rounding raises an
  internal error by name. CUDA is not measured.
