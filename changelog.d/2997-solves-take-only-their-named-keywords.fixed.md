- **`NewtonMFGSolver` stops at the shared non-finite guard before a verdict reads a non-finite change** (#2997). After
  its warm-up and at each Newton iterate, the map's HJB output goes through `diverged_value_function`, so a non-finite
  value function stops the solve with `diverged_nan`, as on every other route, instead of raising the verdict's
  refusal of a non-finite change.
- **A coupling route's `solve()` takes only the keywords its signature names** (#2997, #2922). The solves of
  `FixedPointIterator`, `FictitiousPlayIterator`, the Block iterators and `NewtonMFGSolver` took `**kwargs` and ignored
  what landed there: `relaxation=` and the other damping keywords #2997 removed, `damping=`, `config=` on the Block
  iterators and Newton, and `return_structured=` among them. Each is now Python's own `TypeError`, and so are the five
  names the removal before 1.0 refused by name. `verbose=` is a named parameter of `FixedPointIterator.solve`, which
  ignored it, and of `FictitiousPlayIterator.solve`. On both, a config in force governs progress through its
  `picard.verbose` instead, as it decides the budget and the tolerances; `problem.solve()` writes its own `verbose=`
  there.
