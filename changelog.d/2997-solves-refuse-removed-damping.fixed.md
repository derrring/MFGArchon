- **`NewtonMFGSolver` stops at the shared non-finite guard before a verdict reads a non-finite change** (#2997). After
  its warm-up and at each Newton iterate, the map's HJB output goes through `diverged_value_function`, so a non-finite
  value function stops the solve with `diverged_nan`, as on every other route, instead of raising the verdict's
  refusal of a non-finite change.
- **A coupling route's `solve()` refuses a removed damping keyword by name** (#2997). `FixedPointIterator`,
  `FictitiousPlayIterator`, the Block iterators and `NewtonMFGSolver` keep `**kwargs` in `solve()`, which accepted
  `relaxation=` and the other damping keywords in silence; they now refuse them, naming `damping=`.
