- `reflect_positions`, `absorb_positions` and `wrap_positions` return a (1, d) array for a (1, d) input (#2691).
  - They returned (d,), because the check for a single point ran after `np.atleast_2d`.
  - A (d,) point still comes back as (d,).
  - Before, the obstacle placement of a one-particle step in 2-D or more read its 2-D point as d points of
    dimension 1.
