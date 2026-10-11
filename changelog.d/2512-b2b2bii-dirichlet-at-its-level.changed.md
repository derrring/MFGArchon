- **FEM and meshless Galerkin read a Dirichlet value at the level they solve for** (#2512, row B2; #2538).
  - Before, a Dirichlet value that declares `t`, `g(t, x)` or `g(t)`, was refused at FEM's Dirichlet DOFs, on
    HJB and FP alike, and in meshless Galerkin's Nitsche data load: "has no time to pass it".
  - Each level now reads it at its own time, at each DOF or quadrature point: HJB-FEM and the meshless HJB at
    t_n, FP-FEM at t_{n+1}, on the Picard and Newton steps alike.
  - Measured with the exact solution u = a0 + c t + b.x on every wall, in 1-D and 2-D. For the HJB,
    c = H(b); for the FP, a source c and no drift. At 91f90824 every case was refused. FEM now reproduces it to
    below 2e-15.
  - Meshless Nitsche imposes the wall weakly, so it reproduces no exact solution. Its data load is read at each
    level's time. Its block is still assembled once, and the load is assembled again at each level only when a
    Dirichlet value declares `t`.
  - A constant, or `g(x)`, reads as before.
- **The interpolation applicator reads a face's value through `get_bc_value_at_boundary` alone, and only where the
  face's type uses it** (#2512, row B2). This is HJB-SL's post-step boundary enforcement.
  - It probed the accessor and swallowed `AttributeError`, `ValueError` and `KeyError`. It then fell back to a
    loop over the segments, which applied a segment with no `boundary` to every face. The fallback is gone, and a
    refusal reaches the caller.
  - A BC whose conditions are all zero flux, with a segment restricted to a region, is typed no-flux on every
    face and is no longer read at all.
  - A value on a face whose type does not use it (no-flux, reflecting, periodic, Robin) is no longer read.
    - A constant there was already ignored.
    - A form the owner cannot read there was refused at 91f90824 and is now ignored. For example, a NO_FLUX
      segment with `value=lambda x: 2 * x[0]` raised "declares x … has no point to pass it". It now enforces
      the no-flux wall.
- **Three readers still have no time to pass, and refuse a value that declares `t`:** the FDM calculator route
  (`bc_to_topology_calculator`), `MeshfreeApplicator.apply_field_bc`, and the BC resolver's passthrough
  (`resolve_bc`).
