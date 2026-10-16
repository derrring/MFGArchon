- **The Lions sources refuse a density that is not one slice of their node count, by name** (#2728).
  `create_lions_source` (both paths) and `create_nonlocal_source` read the density as one time-t slice when it holds
  exactly the node count their weights give, or, for one scalar weight, their x. Any other density is refused with
  its shape and with where its node count came from: a `(Nt+1, ...)` trajectory, a stacked multi-population
  density, or a density on other nodes. A trajectory was refused only when it was 2-D. A `(3, 7, 5)` history on a
  7 × 5 grid fell through: the analytic and nonlocal sources raised a shape error naming no trajectory, and the
  finite-difference source with one scalar weight raised nothing and returned a field of 105 values. A density of
  the wrong length did the same on each.
