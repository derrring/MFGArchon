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
    - A Dirichlet dict, a number or a callable, solves bit-for-bit as before in one and two dimensions, and so does a
      no-flux dict, or a Neumann dict with no value or 0, in one.
  - **What changes:**
    - A Neumann dict's `"values"` is its Neumann datum. It was solved as 0 (#2520, under Fixed).
    - In two or more dimensions, a no-flux or Neumann dict gets the face normal at a corner, as a
      `BoundaryConditions` does. A dict got the diagonal one, (±1/√2, ±1/√2), so those solves change near the
      corners (by 6.5e-3 on a 6×6 grid) and now equal the `BoundaryConditions` solve bit for bit.
    - An array of one element is the value. At `bddb45d9` the Dirichlet row read only a zero-dimensional one: a
      one-element 1-D array raised a `TypeError` at the solve, and a Neumann one was dropped with the rest of its
      value (#2520).
    - What a callable returns now meets the boundary-value contract, so `None`, a boolean or more than one number
      is refused. A constant string or boolean `"values"` is refused too, at the solve, where at `bddb45d9` a
      constant `"0.7"` read as 0.7 and `True` as 1. A constant `None` is 0, as before.
  - **What is refused at construction,** each with what it did at `bddb45d9`:
    - a `"type"` that is missing or not a string, which was refused only at the solve: a missing one as "boundary
      conditions required", a `BCType` member with a raw `AttributeError`;
    - a per-point `"values"` (a list, tuple, a dict by point index, or an array of more than one element): a value
      per node is #2600;
    - a nonzero value on a no-flux dict, which was dropped in silence; the solve equalled the plain no-flux one.
      With no value, or 0, a no-flux or periodic dict translates and behaves as before;
    - a nonzero value on a periodic dict, which was refused only at the solve, as every periodic GFDM BC is;
    - a Robin dict, which carried no coefficients and never solved: the row refused its default `alpha = 1`.
      HJB-GFDM's Robin row takes only `alpha = 0, beta = 1`, which is `neumann_bc(value=g)`;
    - an unknown `"type"`, which raised a `ValueError` only at the solve.
  - **Anything else as `boundary_conditions`**, neither a dict nor a `BoundaryConditions` nor `None`, is refused
    with a `TypeError` naming `BoundaryConditions`. Such an object was read only where its attributes matched a
    dict's keys: one with `type="dirichlet"` and `values=g` solved as the dict did, and one with `type="neumann"`
    dropped `g`.
  - **The dict's readers go with it:** the two that called a dict's `"values"` themselves (#2512 row B2's D25 and
    D26), and the `legacy_bc_values` path through the boundary rows. The ghost-particle branch's Dirichlet arm read
    only the dict. Ghost particles are made only at Neumann and no-flux points, so each now takes the mirror value it
    was made for, as it already did for a `BoundaryConditions`.
