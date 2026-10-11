- **`HJBGFDMSolver`'s dict form of `boundary_conditions` is deprecated** (#2597, user ruling; since v0.22.0, recorded
  2026-10-11). A dict such as `{"type": "dirichlet", "values": g}` warns with a `DeprecationWarning` naming
  `BoundaryConditions`, and is translated once, at construction, into the `BoundaryConditions` it spells:
  `dirichlet_bc(value=g)`, `neumann_bc(value=g)`, `no_flux_bc()` or `periodic_bc()`. It becomes removable by the
  deprecation policy at v0.25.0, or six months after 2026-10-11 if that comes first. Pass a `BoundaryConditions`
  instead.
  - **What the translation keeps:**
    - a number is the value, and a missing `"values"` is 0;
    - a callable keeps its old call, by the point alone;
    - the #2513 check of `"value"` against `"values"` is unchanged.
    - A Dirichlet dict, scalar or callable, solves bit-for-bit as before.
  - **What changes:**
    - A Neumann dict's `"values"` is its Neumann datum. It was solved as 0 (#2520, under Fixed).
    - A callable's return now meets the boundary-value contract, so `None`, a boolean or more than one number is
      refused.
  - **What is refused at construction,** each with what it did at `bddb45d9`:
    - a per-point `"values"` (a list, tuple, array, or a dict by point index): a value per node is #2600;
    - a nonzero value on a no-flux dict, which was dropped in silence; the solve equalled the plain no-flux one.
      With no value, or 0, a no-flux or periodic dict translates and behaves as before;
    - a nonzero value on a periodic dict, which was refused only at the solve, as every periodic GFDM BC is;
    - a Robin dict, which carried no coefficients and never solved: the row refused its default `alpha = 1`.
      HJB-GFDM's Robin row takes only `alpha = 0, beta = 1`, which is `neumann_bc(value=g)`;
    - an unknown `"type"`, which raised a `ValueError` only at the solve.
  - **The dict's readers go with it:** the two that called a dict's `"values"` themselves (#2512 row B2's D25 and
    D26), and the `legacy_bc_values` path through the boundary rows. The ghost-particle branch's Dirichlet arm read
    only the dict. Ghost particles are made only at Neumann and no-flux points, so each now takes the mirror value it
    was made for, as it already did for a `BoundaryConditions`.
