- **FEM and meshless Galerkin read a Dirichlet value at the level they solve for** (#2512, row B2; #2538).
  - Before, a Dirichlet value that declares `t`, `g(t, x)` or `g(t)`, was refused at FEM's Dirichlet DOFs, on
    HJB and FP alike, and in meshless Galerkin's Nitsche data load: "has no time to pass it".
  - Each level now reads it at its own time, at each DOF or quadrature point: HJB-FEM and the meshless HJB at
    t_n, FP-FEM at t_{n+1}, on the Picard and Newton steps alike.
  - Measured with the exact solution u = a0 + c t + b.x on every wall, in 1-D and 2-D. For the HJB,
    c = H(b); for the FP, a source c and no drift. At 91f90824 every case was refused. FEM now reproduces it to
    below 2e-15.
  - Meshless Nitsche imposes the wall weakly, so it reproduces no exact solution. Its data load is read at each
    level's time, and the wall holds g(t_n) at each level within 0.006 on the test's fixture.
    - Its block is the same at every level.
    - When a Dirichlet value declares `t`, the Nitsche terms are assembled again at each level, the block
      included. Otherwise they are assembled once.
  - A constant, or `g(x)`, reads as before.
- **The interpolation applicator reads a face's value through `get_bc_value_at_boundary` alone, and only where the
  face's type uses it** (#2512, row B2). This is HJB-SL's post-step boundary enforcement.
  - It probed the accessor and swallowed `AttributeError`, `ValueError` and `KeyError`. It then fell back to a
    loop over the segments, which applied a segment with no `boundary` to every face. The fallback is gone, and a
    refusal reaches the caller.
  - A BC whose conditions are all zero flux, with a segment restricted to a region, is typed no-flux on every
    face and is no longer read at all.
  - A value on a face whose type does not use it (no-flux, reflecting, periodic) is no longer read.
    - A constant there was already ignored.
    - A form the owner cannot read there was refused at 91f90824 and is now ignored. For example, a NO_FLUX
      segment with `value=lambda x: 2 * x[0]` raised "declares x … has no point to pass it". It now enforces
      the no-flux wall.
  - Nor is a value on a Robin face. This applicator's Robin enforcement is a stub: it imposes alpha = beta = 1,
    g = 0 whatever the segment says, as at 91f90824 for a constant value (#2374). HJB-SL refuses a Robin BC
    at construction, so only a direct caller reaches the stub.
- **Four readers still have no time to pass, and refuse a value that declares `t`:**
  - the FDM calculator route (`bc_to_topology_calculator`);
  - `MeshfreeApplicator.apply_field_bc`;
  - the BC resolver's passthrough (`resolve_bc`);
  - FEM's Robin and Neumann load (`assemble_robin_terms`), which stays one constant per segment (#1237).
