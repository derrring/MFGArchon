- **FP-FDM's level 0 is the caller's initial density at every node, a Dirichlet wall's included** (#2984). In 1-D, a
  BC whose segments are all Dirichlet held its uncapped walls at the FP's prescribed value at level 0 in FP-FDM's own
  solve, while the iterators that write `m_initial` back (FixedPoint, FictitiousPlay, the block iterators) held the
  caller's there, so one problem had two level-0 densities. Every other layout, and the FEM, meshless Galerkin and
  network FPs, already held the caller's. The write is deleted, and `initial_density_as_held` returns the caller's
  row. In every other layout nothing moves.
  - Measured with the walls planted at 7.0 on the problem's own BC, `dirichlet_bc(value=0.0)` (1-D, 11 nodes,
    Nt = 5): the direct call's level 0 held 0.0 at the walls at 3f362150, and holds 7.0; the coupled solve's held 7.0
    at both.
  - What reads that level 0, measured against 3f362150 on every coupled route, three BC arrangements, six layouts,
    six coefficients, three couplings, and a fixed and a tolerance stop (#2984's census):
    - **FP-FDM.** Its first step lets an uncapped exit's level-0 content out, and its outflow record counts it; a
      capped exit's queues. The later levels move only through a coefficient that reads the density at a wall: a
      volatility of the whole density, or a local one on an own BC with g ≠ 0, through D·g.
    - **FixedPoint, FictitiousPlay and the block iterators**, which write `m_initial` back: at a fixed iteration
      count they return the same M and U when no coefficient reads the density at a wall. Their residual reads
      FP-FDM's level 0, so a tolerance-stopped solve that never converged now converges. On this issue's fixture it
      did not converge in 50 iterations, with the M residual held at 0.13; it now converges in 21.
    - **Newton, `MultiPopulationIterator`, `GraphMFGSolver` and `RegimeSwitchingIterator`** return or reuse FP-FDM's
      row and write nothing back. Their level 0 is now the caller's density, where it was the prescribed value.
      Their M and U move under a coupling that reads it (a nonlocal one, or a local one where the HJB's walls are not
      Dirichlet), or under a coefficient that reads the density at a wall.
  - **Refused, where the solve ran:**
    - a NaN `M_initial` at such a wall, which is read and refused at FP-FDM's door by name; #2975 read and discarded
      it;
    - a volatility that is non-finite at the caller's wall value, refused at step 1 by name; it read the prescribed
      value there.
  - Level 0 reads no BC value:
    - a wall value g that is non-finite only at t = 0 is not read, since level 0 holds the caller's density, and the
      solve runs, where it was refused by name;
    - a wall no segment covers, with `default_bc` unset, is refused wherever a step reads the BC: by the implicit
      step with #1100's message, by the explicit callable-drift step with `LaplacianOperator`'s. **Stated limit:** a
      solve with no time step reads no BC, so there it returns `m_initial`, where it was refused.
