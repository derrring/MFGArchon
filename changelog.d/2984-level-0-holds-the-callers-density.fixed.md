- **FP-FDM's level 0 is the caller's initial density at every node, a Dirichlet wall's included** (#2984). In 1-D, a
  BC whose segments are all Dirichlet held its walls at g(0) at level 0 on a direct `solve_fp_system` call, while a
  coupled solve's iterators write the caller's `m_initial` over the FP's row, so one problem had two level-0 densities.
  Every other layout, and the FEM, meshless Galerkin and network FPs, already held the caller's. The write is deleted,
  and `initial_density_as_held` returns the caller's row.
  - Measured with the walls planted at 7.0 on the problem's own BC, `dirichlet_bc(value=0.0)` (1-D, 11 nodes,
    Nt = 5): the direct call's level 0 held 0.0 at the walls at 3f362150, and holds 7.0; the coupled solve's held 7.0
    at both.
  - The step's rows hold the prescribed value from level 1 on, so the later levels are unchanged: an exit lets its
    level-0 content out in the first step, and FP-FDM's outflow record counts it.
  - A NaN `M_initial` at such a wall is read, and refused at the door by name; #2975 read and discarded it.
  - Level 0 reads no BC value:
    - a wall value g that is non-finite only at t = 0 is not read, since level 0 holds the caller's density, and the
      solve runs, where it was refused by name;
    - a wall no segment covers, with `default_bc` unset, is refused wherever a step reads the BC, with #1100's
      message unchanged. **Stated limit:** a solve with no time step reads no BC, so there it returns `m_initial`,
      where it was refused.
