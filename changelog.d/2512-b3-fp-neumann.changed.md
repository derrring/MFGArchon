- **An FP solver refuses a NEUMANN handed to it, `g = 0` included** (#2512, row B3). `FPFDMSolver`,
  `FPFVMSolver`, `FPGFDMSolver`, `FPSLSolver` and `FPParticleSolver` accepted
  `boundary_conditions=neumann_bc()` and refused only a nonzero value. A NEUMANN in an FP solver's own BC
  would mean `dm/dn = g`, which no FP solver implements: under a drift, the wall they build is the zero
  total flux `J.n = 0`, a different condition. Pass `no_flux_bc()` for a reflecting FP wall. A
  `default_bc=NEUMANN`, the fall-through for faces no segment names, is refused too, even where no face
  reaches it. The deprecated `mixed_bc` sets it unless told otherwise, so pass
  `default_bc=BCType.NO_FLUX`, or build `BoundaryConditions(segments=...)` directly.
- **One owner reads a shared NEUMANN for every FP solver, and a nonzero value is refused, naming both
  readings** (#2512, row B3). On the problem's shared BC, `NEUMANN(g)` is the HJB's `du/dn = g`. Every FP
  solver reads `NEUMANN(0)` as zero flux (`NO_FLUX`) through `BaseFPSolver`, and refuses `g != 0`. That
  refusal is not new, but its message now says why: `g` says nothing about the agents' mass at the wall,
  and a mass flux through the wall is not a Neumann condition. It also says how to specify the two
  equations' BCs separately: keep `neumann_bc(value=g)` on the problem for the HJB, and pass the FP solver
  `boundary_conditions=no_flux_bc(...)`. `FPFEMSolver` and `MeshlessGalerkinFPSolver` take no BC of their
  own, so they cannot run that model yet; #2532 tracks the route for both. A value that is not provably zero, such
  as a callable, counts as nonzero.
