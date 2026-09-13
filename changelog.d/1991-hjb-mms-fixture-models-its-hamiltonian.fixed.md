- **The 1-D HJB manufactured-solution test measures the scheme, not an unmodelled coupling term**
  (Issue #1991). `TestMMSHJB1D::test_backward_heat_periodic_convergence` solved with a Hamiltonian
  whose `coupling = m` added a first-order error of `T/Nx` that its source never cancelled, on a grid
  ladder (`Nx = 21, 41, 81`) where the error is not yet monotone in `Nx`, and passed its `ratio > 1.5`
  bound at 1.515. It now uses an uncoupled Hamiltonian on `Nx = 81, 161, 321` at `Nt = 20` and bounds
  each error ratio within `(1.6, 2.6)`, measured at 1.96 and 2.07 at `19cbc975`. The upper bound is
  what catches the Godunov branch swap.
