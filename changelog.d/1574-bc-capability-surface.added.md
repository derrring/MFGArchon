- **The full declared BC surface is measured, not trusted** (Issue #1574, phase 1a). Every
  `(solver, BC-type)` pair a solver declares is driven through one solve and checked against the
  invariant that type means: **39 pairs across 11 solvers** (NEUMANN 11, PERIODIC 11, NO_FLUX 11,
  DIRICHLET 4, ROBIN 1, REFLECTING 1). A declaration that the code does not honour, and a
  declaration the code then refuses, are both failures and both now have a number.
  Found: `HJBGFDMSolver` declares `DIRICHLET` and the solve returns **NaN**; `FPGFDMSolver` raises
  for all three types it declares; `FPSLSolver` / `FPSLAdjointSolver` raise at `Nx=81` under
  `NEUMANN` / `NO_FLUX`.
- The oracle is **absolute where the quantity is zero in exact arithmetic** (a Dirichlet wall value,
  a periodic seam) and **monotone convergence over three refinements otherwise** (mass, a normal
  derivative). Three points rather than two, because two are not a trend: `FPSLJacobianSolver`
  improves `1.58e+00 -> 7.64e-03` from Nx=21 to 41 and then gets *worse* at 81, and a two-point
  check certifies it. `HJBFDMSolver` is the opposite case -- `7.42e-01, 6.51e-01, 4.72e-01` is
  genuine slow convergence that a ratio threshold tuned for the fast cases rejects.
- `FPParticleSolver` is **skipped and named** rather than classified: it takes no seed, and over
  three trials of the identical configuration its periodic seam was non-monotone, non-monotone,
  monotone. Marking it xfail asserts a failure it does not reliably have; marking it pass asserts
  the opposite. Seeding it is the fix.
- `ROBIN` and `REFLECTING` are declared once each with no fixture reachable here, and are asserted
  as a **named uncovered set** so they cannot read as covered by being absent.
