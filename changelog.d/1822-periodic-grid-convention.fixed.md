- **A periodic wrap assumed a node layout, and two different ones were in use** (Issues #1822, #1825).
  `TensorProductGrid` builds `np.linspace(lo, hi, N)`, so `x[0]` and `x[-1]` are the same physical
  point; the operator layer (`LaplacianOperator`, `IsotropicDiffusion`) builds
  `np.linspace(lo, hi, N, endpoint=False)`, where every node is distinct. Both reach the same ghost
  applicator, which hard-coded the second — in two separate copies, one per dispatch path.
  On an endpoint-inclusive grid that puts the duplicated node into the ghost and shifts every
  periodic stencil one cell. Measured against the analytic continuation of `sin(2 pi x)` at `Nx=21`:
  the ghost was wrong by **3.09e-01 at ghost depths 1, 2 and 3**. Downstream, `HJBWENOSolver`
  returned a seam of **2.63e-01** from exactly periodic input, and `ImplicitHeatSolver`'s seam
  stalled at ~1.5e-02 under refinement instead of vanishing (#1825) — refinement could not close a
  fixed one-cell offset.
  `PeriodicGridConvention` now names the two layouts and travels on the periodic `BoundaryConditions`,
  which is the one carrier that reaches every wrapping site — a wrap happens *only* because that
  object says PERIODIC — so nothing had to be threaded through the 31 call sites that pad or
  construct. `HJBWENOSolver`'s seam is now exactly **0** and both xfails are gone rather than
  silenced. Closes #1825.
- **Unstated means what this package always did, and a grid completes it** (Issue #1822).
  The convention is `None` by default, and `None` wraps the historical way — all `N` nodes distinct
  — so adding the field changed no existing caller's numbers. `TensorProductGrid` **binds** the
  convention it measures from its own coordinates onto any periodic BC it is handed, the way it
  already binds `dimension`, and refuses one that contradicts them. A BC that never meets a grid is
  completed at the three places that receive both — `get_laplacian_operator`, `get_gradient_operator`
  and the solver BC resolution chain — so no solver declares anything, and a caller with no grid at
  all (the operator layer) keeps the historical layout unless it says otherwise.
  Defaulting to the *other* convention was tried first and reverted under review: it silently
  reinterprets every BC that already meant exclusive, and took the operator layer's own Laplacian
  from **1.3e-02 to 6.3e+02** of error with nothing going red.
- **`LaplacianOperator`'s matrix and its matvec wrapped differently** (Issue #1822).
  `as_scipy_sparse()` hard-coded `(i - 1 + n) % n` while `__call__` padded through the ghost
  applicator. Once the convention was carried on the BC the two paths described different operators,
  disagreeing by **1.9e+02** on an inclusive grid. The sparse assembly now reads the same
  convention; dense and sparse agree to 2.5e-12 under both layouts, pinned by a test that reddens
  when the wrap is reverted — review found the divergence once, then found that fixing it pinned
  nothing.
  Stated limitation: stepping over the duplicated node leaves column `n-1` referenced by nothing but
  its own diagonal, so the inclusive matrix is asymmetric (`|A - A^T| = 900` at `n=31`, against 0.0
  before). No wrong answer follows on seam-consistent input and mass is conserved under the correct
  inclusive trapezoid weights (7e-15), but a future consumer assuming self-adjointness would be
  misled. Filed as #1832; the fix is to carry `n-1` DOFs, as #1829 did for the periodic
  Crank-Nicolson.
- **Two tests were pinning the defect, and are now measured against calculus** (Issue #1822).
  `test_periodic_gradient_byte_identical_to_legacy` asserted that the BC-aware gradient equalled the
  legacy `%Nx` wrap "so periodic baselines do not move". They had to move: that wrap takes the left
  neighbour of `x[0]` to be `x[-1]`, which *is* `x[0]`, so it reads across a separation of `dx` while
  dividing by `2*dx`. Against the analytic `d/dx sin(2 pi x) = 2 pi` at `x=0`: legacy **3.0902**
  (error 3.19), fixed **6.1803** (error 0.10, the interior's own O(h^2)). It now compares to the
  analytic derivative, which cannot go stale when a stencil is replaced. The same one-node error set
  the expected `-1.5` in `test_gradient_uses_periodic_override_not_noflux_geometry`; the correct
  value on its `linspace(0, 4, 5)` grid is `-1.0`, and the periodic-vs-no-flux discrimination the
  test exists for is untouched.
