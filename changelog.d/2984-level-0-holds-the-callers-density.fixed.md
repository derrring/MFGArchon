- **FP-FDM's level 0 is the caller's initial density at every node, a Dirichlet wall's included** (#2984). In 1-D, a
  BC whose segments are all Dirichlet held its uncapped walls at the FP's prescribed value at level 0 in FP-FDM's own
  solve. The iterators that write `m_initial` back held the caller's density there, and the routes that keep FP-FDM's
  row held the prescribed value, so one problem had two level-0 densities. Every other layout, and the FEM, meshless
  Galerkin and network FPs, already held the caller's. The write is deleted, and `initial_density_as_held` returns the
  caller's row. In every other layout nothing moves.
  - Measured with the walls planted at 7.0 on the problem's own BC, `dirichlet_bc(value=0.0)` (1-D, 11 nodes,
    Nt = 5): the direct call's level 0 held 0.0 at the walls at 3f362150, and holds 7.0; the coupled solve's held 7.0
    at both.
  - What reads that level 0, by mechanism; #2984's census measured instances of each against 3f362150:
    - **FP-FDM's own steps** read the caller's value where they read the prescribed value. Its first step lets an
      uncapped exit's level-0 content out, and its outflow record counts it; a capped exit's queues. Every callable
      the step evaluates on the density at step 0 (a volatility, a tensor, a drift) is evaluated at the caller's wall
      value, so a volatility that reads the density there moves the later levels.
    - **FixedPoint, FictitiousPlay, the block iterators and `problem.solve`** write `m_initial` back, so their
      iterate's level 0 is unchanged. Of FP-FDM's returned row only their residual changes an output. So at a fixed
      iteration count M and U change only through FP-FDM's own steps, above, and a tolerance-stopped solve that never
      converged now converges: on this issue's fixture it did not converge in 50 iterations, with the M residual held
      at 0.13, and now converges in 21.
    - **Newton, `MultiPopulationIterator`, `GraphMFGSolver` and `RegimeSwitchingIterator`** keep FP-FDM's row and
      write nothing back. Their level 0 is now the caller's density, where it was the prescribed value. Every consumer
      of it moves (a coupling, the FP velocity of a Hamiltonian whose control reads the density, an HJB source that
      reads it, a coefficient), and their fixed point differs from main's.
    - FP sources are handed the density only at t_{k+1}, so none reads level 0.
  - **Refused, where the solve ran:**
    - a non-finite `M_initial` entry (NaN or +inf) at such a wall, which is read and refused at FP-FDM's door by name;
      #2975 read and discarded it;
    - a callable FP-FDM's first step evaluates on the density that is non-finite or raises at the caller's wall value,
      on every route. It was evaluated at the prescribed value.
  - Level 0 reads no BC value:
    - a wall value g that is non-finite only at t = 0 is not read, since level 0 holds the caller's density, and the
      solve runs, where it was refused by name;
    - a wall no segment covers, with `default_bc` unset, is refused wherever a step reads the BC: by the implicit
      step with #1100's message, by the explicit callable-drift step with `LaplacianOperator`'s. **Stated limit:** a
      solve with no time step reads no BC, so there it returns `m_initial`, where it was refused.
