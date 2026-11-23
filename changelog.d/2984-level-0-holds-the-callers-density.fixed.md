- **FP-FDM's level 0 is the caller's initial density at every node, a Dirichlet wall's included** (#2984). In 1-D, a
  BC whose segments are all Dirichlet held its uncapped walls at the FP's prescribed value at level 0 in FP-FDM's own
  solve, so one problem had two level-0 densities. The iterators that write `m_initial` back held the caller's density
  there; the routes that iterate on FP-FDM's row held the prescribed value or a blend toward it. Every other layout, and
  the FEM, meshless Galerkin and network FPs, already held the caller's. The write is deleted, and
  `initial_density_as_held` returns the caller's row. In every other layout nothing moves.
  - Measured with the walls planted at 7.0 on the problem's own BC, `dirichlet_bc(value=0.0)` (1-D, 11 nodes,
    Nt = 5): the direct call's level 0 held 0.0 at the walls at 3f362150, and holds 7.0; the coupled solve's held 7.0
    at both.
  - **What reads it.** Everything that reads FP-FDM's level 0 now reads the caller's value at those walls:
    - FP-FDM's own steps;
    - the row FP-FDM returns;
    - any array built from that row: the iterate of Newton's phase and of `MultiPopulationIterator`,
      `GraphMFGSolver` and `RegimeSwitchingIterator`, a warm start, and `MFGResidual`'s input.
    The iterators that write `m_initial` back (FixedPoint, FictitiousPlay, the block iterators, `problem.solve`, and
    Newton's Picard warm-up) hold the same level 0 as before after each write-back.
  - **What may change.** Any output computed from those reads may differ, and any refusal or stop that depends on
    such an output may flip, in either direction. Measured instances, not a complete list (#2984's census and
    reviews):
    - the outflow record counts an uncapped exit's level-0 content in the first step; a capped exit's queues;
    - a volatility that reads the density at a wall moves the later levels;
    - the coupled iterators' residual, and with it convergence, adaptive damping and FixedPoint's mass-blowup guard.
      On this issue's fixture the solve did not converge in 50 iterations, with the M residual held at 0.13; it now
      converges in 21.
    - the solution of the routes that iterate on FP-FDM's row, where a coupling, the FP velocity or a coefficient
      carries the wall's value into a row that is not prescribed;
    - a warm start built from FP-FDM's output, and `MFGResidual`.
  - **Flips measured**, where the solve ran before and now stops, or the reverse:
    - a non-finite `M_initial` entry (NaN or +inf) at such a wall is read and refused at FP-FDM's door by name;
      #2975 read and discarded it;
    - a callable FP-FDM's first step evaluates on the density that is non-finite or raises at the caller's wall
      value now fails, on every route;
    - a refusal that depends on the solution's values can fire where the solution moved: the centred scheme's
      negative density on a route that iterates on FP-FDM's row;
    - FixedPoint's mass-blowup guard can stop where it did not, or no longer stop where it did.
  - Level 0 reads no BC value:
    - a wall value g that is non-finite only at t = 0 is not read, since level 0 holds the caller's density, and the
      solve runs, where it was refused by name;
    - a wall no segment covers, with `default_bc` unset, is refused wherever a step reads the BC: by the implicit
      step with #1100's message, by the explicit callable-drift step with `LaplacianOperator`'s. **Stated limit:** a
      solve with no time step reads no BC, so there it returns `m_initial`, where it was refused.
  - **Stated limit:** a wall value near the float limit (1.7e308) makes the outflow record infinite, and nothing
    refuses it.
