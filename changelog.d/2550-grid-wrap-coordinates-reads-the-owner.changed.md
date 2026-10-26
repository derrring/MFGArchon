- **`TensorProductGrid.wrap_coordinates` wraps through the periodic owner, `periodic.wrap_positions`** (#2550, C6-3),
  where it had its own `np.mod`. Each of these behaves differently now:
  - an axis whose span is at most 1e-14, which a grid accepts, is left as it is, the owner's rule for an axis with no
    span; it was wrapped;
  - float32 points against bounds given as Python numbers are wrapped in float64 and rounded once, so the result is
    the exact wrap rounded to float32; the grid's own float32 arithmetic was off by up to 24030 ulps (a sample of
    60000 points, spans up to 1e6);
  - a list of points is accepted, where it raised `AttributeError`;
  - points whose width is not the grid's dimension, a 0-d point on a grid of two or more dimensions, and a
    `(2, N, d)` array with N other than d raise `ValueError`, where they came back wrapped by the grid's own reading
    of their shape, or raised `IndexError`;
  - a 0-d point on a 1-D grid comes back as a `(1, 1)` array, where it raised `IndexError` on a periodic grid and came
    back 0-d on any other; and a `(2, d, d)` array is still read along its middle axis. Shapes outside the documented
    `(N, d)` and `(d,)` are the owners' input contract, #2813.
- **Convention C6 is met**: one owner per wall rule for particles, with a structural guard that fails if a box wall
  rule on positions reappears outside its owner (`docs/user/CONVENTIONS.md` § *Particles at a wall*).
