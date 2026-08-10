- **The U convergence metric ignores an additive function of time** (Issue #1873). `u` reaches the
  density only through `grad(u)` — the drift is `-grad(u)/c` — so adding a constant to a whole time
  slice changes nothing the coupled system can observe. It does change `||u||` and `||du||`, and on
  a real problem it dominates both: measured on the 1-D smoke fixture, **99.77% of `||U||`'s energy
  is that mode**, and 99.71% of the per-sweep change. The error was therefore mostly a quantity
  nothing in the game depends on, in both directions — the mode inflates the absolute error, so a
  converged solve reads as not converged, and it inflates the denominator of the relative error, so
  a solve reads as converged for an unrelated reason: at the sweep where the raw relative error
  first passed `1e-6`, the gauge-free part was still moving at `8.0e-6` and the drift field at
  `9.7e-6`. `M` is deliberately not projected; its level is mass, which is observable and conserved.
  No capability cell changes status; two PASS cells take one to two more Picard sweeps under the
  now-stricter measurement (`fvm_muscl` 18 → 19, `fvm_vs_fdm` 17/18 → 19/19) and the baseline is
  regenerated in the same change. The convergence criterion itself still compares one number
  against both a relative and an absolute error, which is the remaining half of #1873 and is
  deliberately not bundled here — measured, making the arms alternatives while the metric still
  carried the gauge mode reported convergence on a sweep the old form was right to refuse.
