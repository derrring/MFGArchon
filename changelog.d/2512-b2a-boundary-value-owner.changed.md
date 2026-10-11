- **A boundary value is read by one owner, as `g(t, x)`, time first, and a callable is bound by its parameter
  names** (#2512, row B2). `BCSegment.value` and `BoundaryConditions.default_value` were read by more than twenty
  readers, which disagreed:
  - HJB-FDM's ghosts and Dirichlet rows, HJB-SL, HJB-WENO, the 1-D HJB and FP-FDM's interpolation called a callable
    with the time alone.
  - `BoundaryConditions.get_bc_value_at_boundary` and the implicit applicator called it as `(point, time)`, point
    first.
  - FEM and meshless Galerkin called it with the point alone.
  - HJB-GFDM (`BCSegment.get_value`) tried four positional signatures.
  - The BC resolver's passthrough and `bc_to_topology_calculator` read any callable as 0.0.

  Every one now reads through `mfgarchon.geometry.boundary.types.boundary_value`, reached for a segment through
  `segment_value`:
  - **A callable names its parameters `(t, x)`, `(x)` or `(t)`** (`time` is accepted for `t`), and is called by
    name with what it declares. `x` is one point, an array of shape `(d,)`, `d = 1` included, and the callable
    returns one number there.
  - **Refused when the segment is built:**
    - a callable whose parameters are not all named after the slots, such as `lambda s: ...`,
      `lambda t, pos: ...` or `lambda *args: ...`;
    - one that puts `x` before `t`, such as `lambda x, t: ...`, which is never swapped (#2378 phase 5,
      #2375 ruling 8).

    Rename the parameters. This is stricter than the other user callables of #2378 phase 5, which bind parameters
    named otherwise positionally, in the new order: a boundary value is called by name only.
    `MeshfreeApplicator.apply_field_bc`'s `bc_value`, which is not a segment, is bound the same way at first use;
    it called the value positionally with the point, so its parameter must now be named `x`.
  - **Refused at first use, naming the reader and the face:** a callable that declares what its reader does not
    have.
    - FDM reads one value per face, so `x` is refused there. It used to pass the time as `x`: HJB-FDM's ghost
      imposed 5.0 at the exit x = 1 for `lambda x: 10 * x` at t = 0.5.
    - FEM, meshless Galerkin, the FDM calculator route (`bc_to_topology_calculator`),
      `MeshfreeApplicator.apply_field_bc` and the BC resolver's passthrough have no time to pass yet, so `t` is
      refused there.
  - **`BCSegment.get_value`, HJB-GFDM's Dirichlet read, now reads what the callable names.** Measured at the
    point [0.9] and t = 0.5:
    - `(t)` reads the time, 0.5; it read the position, 0.9.
    - `(t, x)` reads 9.5 for `t + 10 * x[0]`. Every signature it tried failed, so it returned 0.0 with a warning.
    - `(t, x)` written unindexed, `t + 10 * x`, reads 9.5. It was called `(point, t)`, swapped, and read 5.9.
    - `(x, t)`, the one form it read correctly, is refused when the segment is built, as above.
    - Its fallback, which returned 0.0 with a warning when no signature fitted, is gone: a callable that cannot be
      bound raises.
  - **A constant reads the same in every reader**, the contract FEM took in #2593:
    - `None`, and whatever `_describe_bc_value` reads as zero, is 0;
    - a finite real number, NumPy scalars included, or a real array of one element, of any shape, is its value;
    - refused: every boolean, every complex value (0 included), an array of more than one element that is not
      all zero, any other type, and a non-finite number.

    At a Dirichlet exit, HJB-FDM's ghost imposed 1.0 for `True`, 0.0 for `np.complex128(1j)` and NaN for `nan`;
    each is now refused. The BC resolver's passthrough read `np.float32(0.7)` and `np.array([0.7])` as 0.0; it now
    reads both as 0.7.
  - **What a callable returns is read as a constant is**, so the owner holds one reading of a value:
    - a return of one element, of any shape, is that element;
    - a longer return is refused, and so are NaN, a boolean and a complex value;
    - `None` is refused, not read as 0, since a function with no `return` gives it.

    At HJB-FDM's Dirichlet wall, a callable returning `nan`, `True` or `np.complex128(1j)` imposed NaN, 1.0 and 0.0.
    On a 2-D face, `lambda t: t * np.arange(1.0, 6.0)` at t = 0.5 was broadcast along the face's padded row, ghost
    corners included, and set the wall's three nodes to 1.0, 1.5 and 2.0. Each is now refused; a field along a face
    is a separate capability.
  - **The FP's shared-Neumann reader refuses a complex value.** It read `np.complex128(1j)` as zero flux, since
    `float()` drops the imaginary part.
  - **A `BCValueProvider`** read by a reader with no iteration state is refused with `NotImplementedError`,
    naming the reader; the coupling layer resolves providers before a solve.
- **The callable binder no longer advises positional binding where it is refused next** (#2512, row B2). For a
  role whose parameters are reordered, time first (a boundary value, a potential), its refusal of a callable
  whose parameters match no slot also offered taking them positionally, in that order, which is refused next.
  It now says to name them, time first.
