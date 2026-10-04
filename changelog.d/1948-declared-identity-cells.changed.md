- **The six silent boundary-applicator cells now apply, refuse, or declare themselves** (Issue #1948, step 2; maintainer rulings 2026-10-04).
  - `GraphApplicator` declares its deliberate no-ops in `_DECLARED_IDENTITY`, each with its reason:
    - DIRICHLET on the density, a value condition (#1471), where `ABSORBING` is the exit that removes density (#1478);
    - SOURCE on the value function;
    - NEUMANN on both fields, which is zero flux on a graph.

    The conformance table holds each declaration to the behaviour in both directions.
  - **A non-zero graph NEUMANN flux is refused** when the `NodeBC` is built (`NotImplementedError`). `apply` dropped it without a word.
  - `InterpolationApplicator` now declares the BC types it enforces and **refuses `EXTRAPOLATION_LINEAR` / `EXTRAPOLATION_QUADRATIC`**. They fell through `enforce_values` with the boundary value left as interpolated. The table recorded them as silent "because the order is already fixed", but `extrapolation_order` is the zero-Neumann extrapolation's order and unrelated.
  - An unrecognised type string in its per-face dispatch now raises instead of being skipped.
