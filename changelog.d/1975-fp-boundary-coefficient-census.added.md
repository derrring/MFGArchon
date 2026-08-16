A test file recording which wall each FP path actually imposes, and which solvers sit outside the
BC capability gate (#1975, #1977, #1979).

Its load-bearing test is an external oracle the area was missing: **mass conservation at a wall
with wall-normal drift.** The conservative schemes -- `divergence_upwind`, the default, and
`divergence_centered` -- impose `J.n = 0` structurally by zeroing the total face flux, conserving
mass to machine precision with `d_n m` nonzero at the wall; the `gradient_*` family imposes
`d_n m = 0` and loses 75-78% (non-conservative by design, #1075). Neither behaviour was asserted
anywhere, and that absence is what let this file's own first version read the right answer as a
defect.

The census now discovers its population with `walk_packages` + `issubclass`, a predicate
independent of the declaration it audits, and records the **ungated** solvers as a population
rather than as an absence: 11 of 22 concrete solvers declare no `_SUPPORTED_BC_TYPES`, including
`FPFEMSolver`, the one FP solver implementing a general Robin. Two further classes apply BCs
without being solver subclasses at all and are named rather than discovered.
