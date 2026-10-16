- **The Lions sources refuse a density that is not one slice of their node count, by name** (#2728).
  `create_lions_source` (both paths) and `create_nonlocal_source` read the density as one time-t slice when it holds
  exactly the node count their weights give, or, for one scalar weight, their x. Any other density is refused with
  its shape: a `(Nt+1, ...)` trajectory, or a density on another grid than the weights'. A trajectory was refused
  only when it was 2-D. A `(3, 7, 5)` history on a 7 × 5 grid fell through to a shape error naming no trajectory,
  and so did a density of the wrong length.
